from __future__ import annotations

from enum import Enum


class OrderStatus(str, Enum):
    LIVE = "live"
    CANCELLED = "cancelled"
    DELIVERED = "delivered"


class Priority(str, Enum):
    NORMAL = "normal"
    RUSH = "rush"


class OrderFilter(str, Enum):
    LIVE = "live"
    UNASSIGNED = "unassigned"
    RUSH = "rush"
    ALL = "all"


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


class ToolCallKey(str, Enum):
    ACTION = "action"
    ARGS = "args"


class Field(str, Enum):
    """Keys of the observation, step result, and view dicts. str-valued so
    json.dumps still emits plain string keys for an LLM agent."""

    # step result envelope
    RESULT = "result"
    ACCEPTED = "accepted"
    NOTE = "note"
    OBSERVATION = "observation"
    # observation
    WAVE = "wave"
    WAVES_TOTAL = "waves_total"
    FINAL_WAVE = "final_wave"
    FEASIBLE = "feasible"
    DEPOT = "depot"
    VEHICLES = "vehicles"
    UNASSIGNED_ORDERS = "unassigned_orders"
    # order and vehicle views
    ID = "id"
    NODE = "node"
    DEMAND = "demand"
    COORD = "coord"
    WINDOW_OPEN = "window_open"
    WINDOW_CLOSE = "window_close"
    PRIORITY = "priority"
    CAPACITY = "capacity"
    IN_SERVICE = "in_service"
    LOAD = "load"
    ASSIGNED = "assigned"
    ROUTE = "route"
    CENTROID = "centroid"
    # tool results
    OBSERVED_TIME = "observed_time"
    VIOLATIONS = "violations"
    TYPE = "type"
    REF = "ref"
    COMMITTED = "committed"
    WAVE_ADVANCED = "wave_advanced"
    REASON = "reason"


class Arg(str, Enum):
    ORDER_ID = "order_id"
    VEHICLE_ID = "vehicle_id"
    STOPS = "stops"
    NODE_A = "a"
    NODE_B = "b"
    FILTER = "filter"
    REASON = "reason"


class DisruptionType(str, Enum):
    BREAKDOWN = "breakdown"
    RUSH_ORDER = "rush_order"
    CANCELLATION = "cancellation"


class ViolationType(str, Enum):
    CAPACITY_EXCEEDED = "capacity_exceeded"
    TIME_WINDOW_MISSED = "time_window_missed"
    SHIFT_END_EXCEEDED = "shift_end_exceeded"
    UNASSIGNED_LIVE_ORDER = "unassigned_live_order"
    ROUTE_MISSING_STOP = "route_missing_stop"
    ROUTE_NOT_DEPOT_ANCHORED = "route_not_depot_anchored"
    OUT_OF_SERVICE_VEHICLE = "out_of_service_vehicle"


class IntegrityFlag(str, Enum):
    INVALID_ACTION_SPAM = "invalid_action_spam"
    NEVER_COMMITTED = "never_committed"
    DISRUPTIONS_UNRESOLVED = "disruptions_unresolved"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
