from __future__ import annotations

from typing import Any

from ..domain import (
    DEPOT,
    ActionType,
    Decision,
    DispatchState,
    Disruption,
    DisruptionKind,
    Order,
    OrderStatus,
    Scenario,
    Trajectory,
    Vehicle,
    is_feasible,
    violations,
)
from .tools import TOOLSET, ToolSpec


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

    def step(self, action: ActionType, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")
        args = args or {}
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
            "feasible": is_feasible(state),
            "vehicles": [self._vehicle_view(v, state) for v in state.vehicles.values()],
            "unassigned_orders": [
                self._order_view(o) for o in state.live_orders() if o.id not in assigned
            ],
        }

    # -- action dispatch ----------------------------------------------------
    def _dispatch_action(self, action: ActionType, args: dict[str, Any]):
        handler = {
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
        }.get(action)
        if handler is None:
            return {}, False, f"unknown action {action}"
        return handler(args)

    def _list_orders(self, args):
        which = args.get("filter", "live")
        if not isinstance(which, str):
            return {}, False, "filter must be a string"
        pool = list(self._state.orders.values())
        assigned = self._state.assigned_ids()
        selected = {
            "all": pool,
            "live": [o for o in pool if o.status is OrderStatus.LIVE],
            "unassigned": [o for o in pool if o.status is OrderStatus.LIVE and o.id not in assigned],
            "rush": [o for o in pool if o.priority.value == "rush"],
        }.get(which, [])
        return [self._order_view(o) for o in selected], True, which

    def _get_vehicle(self, args):
        vehicle = self._vehicle(args.get("vehicle_id"))
        if vehicle is None:
            return {}, False, "no such vehicle"
        return self._vehicle_view(vehicle, self._state), True, ""

    def _query_traffic(self, args):
        node_a, node_b = self._as_node(args.get("a")), self._as_node(args.get("b"))
        if node_a is None or node_b is None:
            return {}, False, "node out of range"
        return {"true_time": float(self._state.network.true_time[node_a, node_b])}, True, ""

    def _check_feasibility(self, args):
        found = violations(self._state)
        return {
            "feasible": not found,
            "violations": [{"kind": v.kind.value, "ref": v.ref} for v in found],
        }, True, ""

    def _assign_order(self, args):
        order = self._order(args.get("order_id"))
        vehicle = self._vehicle(args.get("vehicle_id"))
        if order is None or vehicle is None:
            return {}, False, "unknown order or vehicle"
        if order.status is not OrderStatus.LIVE:
            return {}, False, "order not live"
        if not vehicle.in_service:
            return {}, False, "vehicle out of service"
        self._detach(order.id)
        vehicle.assigned.append(order.id)
        return {}, True, ""

    def _unassign_order(self, args):
        return {}, self._detach(self._key(args.get("order_id"))), ""

    def _set_route(self, args):
        vehicle = self._vehicle(args.get("vehicle_id"))
        stops = args.get("stops")
        if vehicle is None or not isinstance(stops, list):
            return {}, False, "unknown vehicle or bad stops"
        parsed = [self._as_node(stop) for stop in stops]
        if any(node is None for node in parsed):
            return {}, False, "stop out of range or non-integer"
        vehicle.route = parsed
        return {}, True, ""

    def _reroute(self, args):
        vehicle = self._vehicle(args.get("vehicle_id"))
        if vehicle is None:
            return {}, False, "unknown vehicle"
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
        return {"reason": args.get("reason", "")}, True, "refused"

    # -- mutation helpers ---------------------------------------------------
    @staticmethod
    def _key(value) -> str | None:
        return value if isinstance(value, str) else None

    def _vehicle(self, value) -> Vehicle | None:
        return self._state.vehicles.get(self._key(value))

    def _order(self, value) -> Order | None:
        return self._state.orders.get(self._key(value))

    def _as_node(self, value) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value if 0 <= value < self._state.network.size else None

    def _detach(self, order_id: str | None) -> bool:
        removed = False
        for vehicle in self._state.vehicles.values():
            if order_id in vehicle.assigned:
                vehicle.assigned.remove(order_id)
                removed = True
        return removed

    def _apply(self, disruption: Disruption) -> None:
        if disruption.kind is DisruptionKind.BREAKDOWN:
            vehicle = self._state.vehicles[disruption.vehicle_id]
            vehicle.in_service = False
            for order_id in list(vehicle.assigned):
                self._detach(order_id)
            vehicle.route = []
        elif disruption.kind is DisruptionKind.RUSH_ORDER:
            self._state.orders[disruption.order.id] = disruption.order
        elif disruption.kind is DisruptionKind.CANCELLATION:
            order = self._state.orders.get(disruption.order_id)
            if order is not None:
                self._detach(order.id)
                self._state.orders[order.id] = order.with_status(OrderStatus.CANCELLED)

    def _nearest_route(self, vehicle: Vehicle) -> list[int]:
        targets = [self._state.orders[o].node for o in vehicle.assigned]
        route = [DEPOT]
        remaining = list(dict.fromkeys(targets))
        times = self._state.network.true_time
        while remaining:
            here = route[-1]
            nxt = min(remaining, key=lambda node: times[here, node])
            route.append(nxt)
            remaining.remove(nxt)
        route.append(DEPOT)
        return route

    # -- views --------------------------------------------------------------
    @staticmethod
    def _order_view(order: Order) -> dict[str, Any]:
        return {
            "id": order.id, "node": order.node, "demand": order.demand,
            "window_open": order.window_open, "window_close": order.window_close,
            "priority": order.priority.value,
        }

    @staticmethod
    def _vehicle_view(vehicle: Vehicle, state: DispatchState) -> dict[str, Any]:
        return {
            "id": vehicle.id, "capacity": vehicle.capacity, "in_service": vehicle.in_service,
            "load": vehicle.load(state.orders), "assigned": list(vehicle.assigned),
            "route": list(vehicle.route),
        }
