from __future__ import annotations

from dataclasses import dataclass

from optimbench.domain.enums import ActionType, Arg


@dataclass(frozen=True)
class ToolSpec:
    action: ActionType
    args: tuple[Arg, ...]
    summary: str


TOOLSET: tuple[ToolSpec, ...] = (
    ToolSpec(ActionType.LIST_ORDERS, (Arg.FILTER,), "List orders (filter: live | unassigned | rush | all)."),
    ToolSpec(ActionType.GET_VEHICLE, (Arg.VEHICLE_ID,), "Inspect one vehicle's load, capacity and route."),
    ToolSpec(ActionType.QUERY_TRAFFIC, (Arg.NODE_A, Arg.NODE_B), "Ground-truth travel time between two nodes."),
    ToolSpec(ActionType.CHECK_FEASIBILITY, (), "Dry-run the hard constraints on the current plan."),
    ToolSpec(ActionType.ASSIGN_ORDER, (Arg.ORDER_ID, Arg.VEHICLE_ID), "Assign an order to a vehicle."),
    ToolSpec(ActionType.UNASSIGN_ORDER, (Arg.ORDER_ID,), "Remove an order from its vehicle."),
    ToolSpec(ActionType.SET_ROUTE, (Arg.VEHICLE_ID, Arg.STOPS), "Set a vehicle's explicit stop sequence."),
    ToolSpec(ActionType.REROUTE, (Arg.VEHICLE_ID,), "Auto-sequence a vehicle's assigned stops from the depot."),
    ToolSpec(ActionType.DISPATCH, (), "Submit the current plan for this wave (required to score it) and advance; the final wave must be dispatched too."),
    ToolSpec(ActionType.REFUSE, (Arg.REASON,), "Decline a request that cannot be satisfied."),
)
