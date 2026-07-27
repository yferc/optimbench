"""A near-exact optimum for the dispatch objective, via OR-Tools CVRP.

This is the yardstick the heuristic reference is measured against, offline. It solves the same
quantity the task score's denominator approximates: the minimum total true travel time to serve
every live order with the in-service fleet, respecting capacity, each vehicle a depot-anchored
tour. It is never used in scoring, only to report how tight the heuristic reference is.
"""
from __future__ import annotations

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from optimbench.domain import DEPOT, DispatchState

_SCALE = 1000  # OR-Tools arc costs are integers; travel times are scaled then unscaled


def optimal_cost(state: DispatchState, time_limit_s: int = 2) -> float:
    """Minimum total true travel time to serve the live orders with the in-service fleet.

    A capacitated vehicle routing solve (guided local search, time-boxed), comparable to
    fleet_cost and reference_cost. Raises if no feasible routing is found in the time budget.
    """
    vehicles = [v for v in state.vehicles.values() if v.in_service]
    if not vehicles:
        return 0.0

    demand_by_node: dict[int, int] = {}
    for order in state.live_orders():
        node = order.node
        demand_by_node[node] = (demand_by_node[node] if node in demand_by_node else 0) + order.demand
    if not demand_by_node:
        return 0.0

    locations = [DEPOT, *sorted(demand_by_node)]
    demands = [0, *(demand_by_node[node] for node in locations[1:])]
    times = state.network.true_time

    manager = pywrapcp.RoutingIndexManager(len(locations), len(vehicles), 0)
    routing = pywrapcp.RoutingModel(manager)

    def arc_cost(from_index: int, to_index: int) -> int:
        a, b = locations[manager.IndexToNode(from_index)], locations[manager.IndexToNode(to_index)]
        return round(float(times[a, b]) * _SCALE)

    transit = routing.RegisterTransitCallback(arc_cost)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    def demand_at(index: int) -> int:
        return demands[manager.IndexToNode(index)]

    demand_callback = routing.RegisterUnaryTransitCallback(demand_at)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback, 0, [v.capacity for v in vehicles], True, "Capacity"
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit_s)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools found no feasible routing within the time limit")
    return solution.ObjectiveValue() / _SCALE
