from __future__ import annotations

from typing import Any

from optimbench.domain import (
    DEPOT,
    ActionType,
    Arg,
    Decision,
    DispatchState,
    Disruption,
    DisruptionType,
    Order,
    OrderFilter,
    OrderStatus,
    Priority,
    Scenario,
    Trajectory,
    Vehicle,
    is_feasible,
    violations,
)
from optimbench.simulation.tools import TOOLSET, ToolSpec

_FILTERS = {member.value: member for member in OrderFilter}


class DispatchEnvironment:
    def __init__(self, max_turns_per_wave: int = 80) -> None:
        self._max_turns_per_wave = max_turns_per_wave
        self._scenario: Scenario | None = None
        self._state: DispatchState | None = None
        self._trajectory = Trajectory()
        self._turn = 0
        self._wave_cursor = 0
        self._done = False

    def reset(self, scenario: Scenario) -> dict[str, Any]:
        self._scenario = scenario
        self._state = scenario.state
        self._trajectory = Trajectory()
        self._turn = 0
        self._wave_cursor = 0
        self._done = False
        return self.observation()

    def tools(self) -> tuple[ToolSpec, ...]:
        return TOOLSET

    def step(self, action: ActionType, args: dict[Arg, Any]) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")
        result, accepted, note = self._dispatch_action(action, args)
        self._trajectory.record(Decision(self._turn, action, args, accepted, note))
        self._turn += 1
        if self._turn > self._max_turns_per_wave * (len(self._scenario.disruptions) + 1):
            self._done = True
        return {"result": result, "accepted": accepted, "note": note, "observation": self.observation()}

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

    # -- observation --------------------------------------------------------
    def observation(self) -> dict[str, Any]:
        state = self._state
        assigned = state.assigned_ids()
        return {
            "wave": state.wave,
            "waves_total": len(self._scenario.disruptions),
            "final_wave": self._wave_cursor >= len(self._scenario.disruptions),
            "feasible": is_feasible(state),
            "depot": _coord(state, DEPOT),
            "vehicles": [self._vehicle_view(v, state) for v in state.vehicles.values()],
            "unassigned_orders": [
                self._order_view(o, state) for o in state.live_orders() if o.id not in assigned
            ],
        }

    # -- action dispatch ----------------------------------------------------
    def _dispatch_action(self, action: ActionType, args: dict[Arg, Any]):
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
        }[action](args)

    def _list_orders(self, args):
        name = args[Arg.FILTER]
        if name not in _FILTERS:
            return {}, False, "unknown filter"
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
        return [self._order_view(o, self._state) for o in selected], True, which.value

    def _get_vehicle(self, args):
        vehicle_id = args[Arg.VEHICLE_ID]
        if vehicle_id not in self._state.vehicles:
            return {}, False, "no such vehicle"
        return self._vehicle_view(self._state.vehicles[vehicle_id], self._state), True, ""

    def _query_traffic(self, args):
        a, b = args[Arg.NODE_A], args[Arg.NODE_B]
        if not (self._in_bounds(a) and self._in_bounds(b)):
            return {}, False, "node out of range"
        return {"true_time": float(self._state.network.true_time[a, b])}, True, ""

    def _check_feasibility(self, args):
        found = violations(self._state)
        return {
            "feasible": not found,
            "violations": [{"type": v.type.value, "ref": v.ref} for v in found],
        }, True, ""

    def _assign_order(self, args):
        order_id, vehicle_id = args[Arg.ORDER_ID], args[Arg.VEHICLE_ID]
        if order_id not in self._state.orders or vehicle_id not in self._state.vehicles:
            return {}, False, "unknown order or vehicle"
        order, vehicle = self._state.orders[order_id], self._state.vehicles[vehicle_id]
        if order.status is not OrderStatus.LIVE:
            return {}, False, "order not live"
        if not vehicle.in_service:
            return {}, False, "vehicle out of service"
        self._detach(order_id)
        vehicle.assigned.append(order_id)
        return {}, True, ""

    def _unassign_order(self, args):
        return {}, self._detach(args[Arg.ORDER_ID]), ""

    def _set_route(self, args):
        vehicle_id, stops = args[Arg.VEHICLE_ID], args[Arg.STOPS]
        if vehicle_id not in self._state.vehicles:
            return {}, False, "unknown vehicle"
        if any(not self._in_bounds(node) for node in stops):
            return {}, False, "stop out of range"
        self._state.vehicles[vehicle_id].route = list(stops)
        return {}, True, ""

    def _reroute(self, args):
        vehicle_id = args[Arg.VEHICLE_ID]
        if vehicle_id not in self._state.vehicles:
            return {}, False, "unknown vehicle"
        vehicle = self._state.vehicles[vehicle_id]
        vehicle.route = self._nearest_route(vehicle)
        return {"route": vehicle.route}, True, ""

    def _commit(self, args):
        if self._wave_cursor < len(self._scenario.disruptions):
            self._apply(self._scenario.disruptions[self._wave_cursor])
            self._wave_cursor += 1
            self._state.wave += 1
            return {"committed": True, "wave_advanced": True}, True, "disruption applied"
        self._done = True
        return {"committed": True, "wave_advanced": False}, True, "final commit"

    def _refuse(self, args):
        return {"reason": args[Arg.REASON]}, True, "refused"

    # -- mutation helpers ---------------------------------------------------
    def _in_bounds(self, node: int) -> bool:
        return 0 <= node < self._state.network.size

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
    def _order_view(order: Order, state: DispatchState) -> dict[str, Any]:
        return {
            "id": order.id, "node": order.node, "demand": order.demand,
            "coord": _coord(state, order.node),
            "window_open": order.window_open, "window_close": order.window_close,
            "priority": order.priority.value,
        }

    @staticmethod
    def _vehicle_view(vehicle: Vehicle, state: DispatchState) -> dict[str, Any]:
        return {
            "id": vehicle.id, "capacity": vehicle.capacity, "in_service": vehicle.in_service,
            "load": vehicle.load(state.orders), "assigned": list(vehicle.assigned),
            "route": list(vehicle.route),
            "centroid": _load_centroid(vehicle, state),
        }


def _coord(state: DispatchState, node: int) -> list[float]:
    return [float(c) for c in state.network.coordinates[node]]


def _load_centroid(vehicle: Vehicle, state: DispatchState) -> list[float]:
    nodes = [state.orders[o].node for o in vehicle.assigned if o in state.orders]
    if not nodes:
        return _coord(state, DEPOT)
    points = [state.network.coordinates[node] for node in nodes]
    return [sum(float(p[i]) for p in points) / len(points) for i in (0, 1)]
