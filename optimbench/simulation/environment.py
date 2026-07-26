"""The dispatch environment: a gym-style loop over the agent tool API."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimbench.domain import (
    DEPOT,
    TOOLSET,
    ActionType,
    Arg,
    Decision,
    DispatchState,
    Disruption,
    DisruptionType,
    Field,
    Note,
    Order,
    OrderFilter,
    OrderStatus,
    Priority,
    Scenario,
    ToolSpec,
    Trajectory,
    Vehicle,
    is_feasible,
    violations,
)

_FILTERS = {member.value: member for member in OrderFilter}


@dataclass(frozen=True)
class ActionOutcome:
    accepted: bool
    note: Note = Note.NONE
    result: dict[Field, Any] = field(default_factory=dict)


class DispatchEnvironment:
    """Gym-style environment exposing the dispatch tool API.

    reset(scenario) returns the first observation. step(action, args) applies one tool
    call and returns a dict keyed by the Field enum: the result, whether it was accepted,
    an explanatory note, the new observation, and the terminated and truncated flags. The
    dispatch action commits the current wave and applies the next disruption; read actions
    return information and mutation actions edit the plan. Terminated marks the final
    commit; truncated marks the per-wave turn cap. The environment holds no RNG, so a
    scenario replays identically.
    """

    def __init__(self, max_turns_per_wave: int = 80) -> None:
        self._max_turns_per_wave = max_turns_per_wave
        self._scenario: Scenario | None = None
        self._state: DispatchState | None = None
        self._trajectory = Trajectory()
        self._turn = 0
        self._wave_turns = 0
        self._wave_cursor = 0
        self._done = False
        self._truncated = False

    def reset(self, scenario: Scenario) -> dict[Field, Any]:
        self._scenario = scenario
        self._state = scenario.state
        self._trajectory = Trajectory()
        self._turn = 0
        self._wave_turns = 0
        self._wave_cursor = 0
        self._done = False
        self._truncated = False
        return self.observation()

    def tools(self) -> tuple[ToolSpec, ...]:
        return TOOLSET

    def step(self, action: ActionType, args: dict[Arg, Any]) -> dict[Field, Any]:
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")
        outcome = self._route_action(action, args)
        self._trajectory.record(Decision(self._turn, action, args, outcome.accepted, outcome.note))
        self._turn += 1
        self._wave_turns += 1
        if action is ActionType.DISPATCH and outcome.accepted:
            self._wave_turns = 0  # a committed wave resets the per-wave turn budget
        if not self._done and self._wave_turns > self._max_turns_per_wave:
            self._done = True
            self._truncated = True
        return {
            Field.RESULT: outcome.result, Field.ACCEPTED: outcome.accepted,
            Field.NOTE: outcome.note, Field.OBSERVATION: self.observation(),
            Field.TERMINATED: self.terminated, Field.TRUNCATED: self._truncated,
        }

    @property
    def state(self) -> DispatchState:
        return self._state

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def done(self) -> bool:
        return self._done

    @property
    def terminated(self) -> bool:
        """True when the episode ended by committing the final wave (a real MDP terminal)."""
        return self._done and not self._truncated

    @property
    def truncated(self) -> bool:
        """True when the episode ended by hitting the per-wave turn cap, not by finishing."""
        return self._truncated

    # -- observation --------------------------------------------------------
    def observation(self) -> dict[Field, Any]:
        state = self._state
        assigned = state.assigned_ids()
        return {
            Field.WAVE: state.wave,
            Field.WAVES_TOTAL: len(self._scenario.disruptions),
            Field.FINAL_WAVE: self._wave_cursor >= len(self._scenario.disruptions),
            Field.FEASIBLE: is_feasible(state),
            Field.DEPOT: _coord(state, DEPOT),
            Field.VEHICLES: [self._vehicle_view(v, state) for v in state.vehicles.values()],
            Field.UNASSIGNED_ORDERS: [
                self._order_view(o, state) for o in state.live_orders() if o.id not in assigned
            ],
        }

    # -- action routing -----------------------------------------------------
    def _route_action(self, action: ActionType, args: dict[Arg, Any]) -> ActionOutcome:
        return {
            ActionType.LIST_ORDERS: self._list_orders,
            ActionType.GET_VEHICLE: self._get_vehicle,
            ActionType.QUERY_TRAFFIC: self._query_traffic,
            ActionType.CHECK_FEASIBILITY: self._check_feasibility,
            ActionType.ASSIGN_ORDER: self._assign_order,
            ActionType.UNASSIGN_ORDER: self._unassign_order,
            ActionType.SET_ROUTE: self._set_route,
            ActionType.REROUTE: self._reroute,
            ActionType.DISPATCH: self._commit,
            ActionType.REFUSE: self._refuse,
            ActionType.INVALID: self._invalid,
        }[action](args)

    def _invalid(self, args) -> ActionOutcome:
        return ActionOutcome(False, Note.MALFORMED_TOOL_CALL)

    def _list_orders(self, args) -> ActionOutcome:
        name = args[Arg.FILTER]
        if name not in _FILTERS:
            return ActionOutcome(False, Note.UNKNOWN_FILTER)
        which = _FILTERS[name]
        pool = list(self._state.orders.values())
        assigned = self._state.assigned_ids()
        selected = {
            OrderFilter.ALL: pool,
            OrderFilter.LIVE: [o for o in pool if o.status is OrderStatus.LIVE],
            OrderFilter.UNASSIGNED:
                [o for o in pool if o.status is OrderStatus.LIVE and o.id not in assigned],
            OrderFilter.RUSH: [o for o in pool if o.priority is Priority.RUSH],
        }[which]
        return ActionOutcome(True, result={Field.ORDERS:
                                            [self._order_view(o, self._state) for o in selected]})

    def _get_vehicle(self, args) -> ActionOutcome:
        vehicle_id = args[Arg.VEHICLE_ID]
        if vehicle_id not in self._state.vehicles:
            return ActionOutcome(False, Note.UNKNOWN_VEHICLE)
        return ActionOutcome(True, result=self._vehicle_view(self._state.vehicles[vehicle_id], self._state))

    def _query_traffic(self, args) -> ActionOutcome:
        node_a, node_b = args[Arg.NODE_A], args[Arg.NODE_B]
        if not (self._in_bounds(node_a) and self._in_bounds(node_b)):
            return ActionOutcome(False, Note.NODE_OUT_OF_RANGE)
        travel = float(self._state.network.observed_time[node_a, node_b])
        return ActionOutcome(True, result={Field.OBSERVED_TIME: travel})

    def _check_feasibility(self, args) -> ActionOutcome:
        found = violations(self._state)
        return ActionOutcome(True, result={
            Field.FEASIBLE: not found,
            Field.VIOLATIONS: [{Field.TYPE: v.type, Field.REF: v.ref} for v in found],
        })

    def _assign_order(self, args) -> ActionOutcome:
        order_id, vehicle_id = args[Arg.ORDER_ID], args[Arg.VEHICLE_ID]
        if order_id not in self._state.orders or vehicle_id not in self._state.vehicles:
            return ActionOutcome(False, Note.UNKNOWN_ORDER_OR_VEHICLE)
        order, vehicle = self._state.orders[order_id], self._state.vehicles[vehicle_id]
        if order.status is not OrderStatus.LIVE:
            return ActionOutcome(False, Note.ORDER_NOT_LIVE)
        if not vehicle.in_service:
            return ActionOutcome(False, Note.VEHICLE_OUT_OF_SERVICE)
        self._detach(order_id)
        vehicle.assigned.append(order_id)
        return ActionOutcome(True)

    def _unassign_order(self, args) -> ActionOutcome:
        return ActionOutcome(self._detach(args[Arg.ORDER_ID]))

    def _set_route(self, args) -> ActionOutcome:
        vehicle_id, stops = args[Arg.VEHICLE_ID], args[Arg.STOPS]
        if vehicle_id not in self._state.vehicles:
            return ActionOutcome(False, Note.UNKNOWN_VEHICLE)
        if any(not self._in_bounds(node) for node in stops):
            return ActionOutcome(False, Note.STOP_OUT_OF_RANGE)
        self._state.vehicles[vehicle_id].route = list(stops)
        return ActionOutcome(True)

    def _reroute(self, args) -> ActionOutcome:
        vehicle_id = args[Arg.VEHICLE_ID]
        if vehicle_id not in self._state.vehicles:
            return ActionOutcome(False, Note.UNKNOWN_VEHICLE)
        vehicle = self._state.vehicles[vehicle_id]
        vehicle.route = self._nearest_route(vehicle)
        return ActionOutcome(True, result={Field.ROUTE: list(vehicle.route)})

    def _commit(self, args) -> ActionOutcome:
        if self._wave_cursor < len(self._scenario.disruptions):
            self._apply(self._scenario.disruptions[self._wave_cursor])
            self._wave_cursor += 1
            self._state.wave += 1
            return ActionOutcome(True, Note.DISRUPTION_APPLIED,
                                 {Field.COMMITTED: True, Field.WAVE_ADVANCED: True})
        self._done = True
        return ActionOutcome(True, Note.FINAL_COMMIT,
                             {Field.COMMITTED: True, Field.WAVE_ADVANCED: False})

    def _refuse(self, args) -> ActionOutcome:
        return ActionOutcome(True, Note.REFUSED, {Field.REASON: args[Arg.REASON]})

    # -- mutation helpers ---------------------------------------------------
    def _in_bounds(self, node: int) -> bool:
        # bool is an int subclass; reject True/False so they cannot index the matrix as 0/1
        return isinstance(node, int) and not isinstance(node, bool) and 0 <= node < self._state.network.size

    def _detach(self, order_id: str) -> bool:
        removed = False
        for vehicle in self._state.vehicles.values():
            if order_id in vehicle.assigned:
                vehicle.assigned.remove(order_id)
                removed = True
        return removed

    def _busiest_vehicle(self) -> Vehicle | None:
        candidates = [v for v in self._state.vehicles.values() if v.in_service and v.assigned]
        if not candidates:
            return None
        # id breaks load ties so the breakdown target is deterministic across replays
        return max(candidates, key=lambda v: (v.load(self._state.orders), v.id))

    def _apply(self, disruption: Disruption) -> None:
        if disruption.type is DisruptionType.BREAKDOWN:
            vehicle = self._busiest_vehicle()
            if vehicle is not None:
                vehicle.in_service = False
                for order_id in list(vehicle.assigned):
                    self._detach(order_id)
                vehicle.route = []
        elif disruption.type is DisruptionType.RUSH_ORDER:
            self._state.orders[disruption.order.id] = disruption.order
        elif disruption.type is DisruptionType.CANCELLATION:
            order = self._state.orders[disruption.order_id]
            self._detach(order.id)
            self._state.orders[order.id] = order.with_status(OrderStatus.CANCELLED)

    def _nearest_route(self, vehicle: Vehicle) -> list[int]:
        targets = list(dict.fromkeys(self._state.orders[o].node for o in vehicle.assigned))
        times = self._state.network.true_time
        route = [DEPOT]
        while targets:
            here = route[-1]
            nearest = min(targets, key=lambda node: times[here, node])
            route.append(nearest)
            targets.remove(nearest)
        route.append(DEPOT)
        return route

    # -- views --------------------------------------------------------------
    @staticmethod
    def _order_view(order: Order, state: DispatchState) -> dict[Field, Any]:
        return {
            Field.ID: order.id, Field.NODE: order.node, Field.DEMAND: order.demand,
            Field.COORD: _coord(state, order.node),
            Field.WINDOW_OPEN: order.window_open, Field.WINDOW_CLOSE: order.window_close,
            Field.PRIORITY: order.priority,
        }

    @staticmethod
    def _vehicle_view(vehicle: Vehicle, state: DispatchState) -> dict[Field, Any]:
        return {
            Field.ID: vehicle.id, Field.CAPACITY: vehicle.capacity,
            Field.IN_SERVICE: vehicle.in_service, Field.LOAD: vehicle.load(state.orders),
            Field.ASSIGNED: list(vehicle.assigned), Field.ROUTE: list(vehicle.route),
            Field.CENTROID: _load_centroid(vehicle, state),
        }


def _coord(state: DispatchState, node: int) -> list[float]:
    return [float(c) for c in state.network.coordinates[node]]


def _load_centroid(vehicle: Vehicle, state: DispatchState) -> list[float]:
    nodes = [state.orders[o].node for o in vehicle.assigned]
    if not nodes:
        return _coord(state, DEPOT)
    points = [state.network.coordinates[node] for node in nodes]
    return [sum(float(p[i]) for p in points) / len(points) for i in (0, 1)]
