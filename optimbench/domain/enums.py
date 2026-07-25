from __future__ import annotations

from enum import Enum


class OrderStatus(str, Enum):
    LIVE = "live"
    CANCELLED = "cancelled"
    DELIVERED = "delivered"


class Priority(str, Enum):
    NORMAL = "normal"
    RUSH = "rush"


class ActionType(str, Enum):
    LIST_ORDERS = "list_orders"
    GET_VEHICLE = "get_vehicle"
    QUERY_TRAFFIC = "query_traffic"
    CHECK_FEASIBILITY = "check_feasibility"
    ASSIGN_ORDER = "assign_order"
    UNASSIGN_ORDER = "unassign_order"
    SET_ROUTE = "set_route"
    REROUTE = "reroute"
    DISPATCH = "dispatch"
    REFUSE = "refuse"


class DisruptionKind(str, Enum):
    BREAKDOWN = "breakdown"
    RUSH_ORDER = "rush_order"
    CANCELLATION = "cancellation"


class ViolationKind(str, Enum):
    CAPACITY_EXCEEDED = "capacity_exceeded"
    TIME_WINDOW_MISSED = "time_window_missed"
    UNASSIGNED_LIVE_ORDER = "unassigned_live_order"
    ROUTE_MISSING_STOP = "route_missing_stop"
    OUT_OF_SERVICE_VEHICLE = "out_of_service_vehicle"


class IntegrityFlag(str, Enum):
    INVALID_ACTION_SPAM = "invalid_action_spam"
    ACCEPTED_INFEASIBLE_RUSH = "accepted_infeasible_rush"
    NEVER_COMMITTED = "never_committed"


class ReferenceKind(str, Enum):
    OPTIMAL = "optimal"
    HEURISTIC = "heuristic"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
