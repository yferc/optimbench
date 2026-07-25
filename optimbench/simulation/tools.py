from __future__ import annotations

from dataclasses import dataclass

from ..domain import ActionType


@dataclass(frozen=True)
class ToolSpec:
    action: ActionType
    args: tuple[str, ...]
    summary: str


TOOLSET: tuple[ToolSpec, ...] = (
    ToolSpec(ActionType.LIST_ORDERS, ("filter",), "List orders (filter: live | unassigned | rush | all)."),
    ToolSpec(ActionType.GET_VEHICLE, ("vehicle_id",), "Inspect one vehicle's load, capacity and route."),
    ToolSpec(ActionType.QUERY_TRAFFIC, ("a", "b"), "Ground-truth travel time between two nodes."),
    ToolSpec(ActionType.CHECK_FEASIBILITY, (), "Dry-run the hard constraints on the current plan."),
    ToolSpec(ActionType.ASSIGN_ORDER, ("order_id", "vehicle_id"), "Assign an order to a vehicle."),
    ToolSpec(ActionType.UNASSIGN_ORDER, ("order_id",), "Remove an order from its vehicle."),
    ToolSpec(ActionType.SET_ROUTE, ("vehicle_id", "stops"), "Set a vehicle's explicit stop sequence."),
    ToolSpec(ActionType.REROUTE, ("vehicle_id",), "Auto-sequence a vehicle's assigned stops from the depot."),
    ToolSpec(ActionType.DISPATCH, (), "Submit the current plan for this wave (required to score it) and advance; the final wave must be dispatched too."),
    ToolSpec(ActionType.REFUSE, ("reason",), "Decline a request that cannot be satisfied."),
)

TOOLSET_BY_ACTION: dict[ActionType, ToolSpec] = {spec.action: spec for spec in TOOLSET}
