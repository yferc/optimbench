from __future__ import annotations

import numpy as np
import pygame

from ..domain import DEPOT, DispatchState, OrderStatus, Priority, fleet_cost, is_feasible

_BG = (13, 19, 32)
_ARENA = (26, 34, 52)
_DEPOT = (75, 212, 138)
_LIVE_ASSIGNED = (46, 230, 200)
_UNASSIGNED = (255, 182, 74)
_RUSH = (255, 92, 140)
_CANCELLED = (90, 100, 120)
_TEXT = (214, 228, 240)
_ROUTE = (
    (46, 230, 200), (255, 182, 74), (169, 155, 255),
    (120, 240, 255), (255, 128, 120), (150, 255, 170),
)
_WORLD = 100.0


class EpisodeRenderer:
    def __init__(self, size: int = 720, margin: int = 64) -> None:
        pygame.init()
        pygame.font.init()
        self._size = size
        self._margin = margin
        self._font = pygame.font.SysFont("Menlo,Consolas,monospace", 20, bold=True)
        self._small = pygame.font.SysFont("Menlo,Consolas,monospace", 15, bold=True)
        self._surface = pygame.Surface((size, size))

    def frame(self, state: DispatchState, waves_total: int) -> np.ndarray:
        self._surface.fill(_BG)
        pygame.draw.rect(self._surface, _ARENA,
                         (self._margin, self._margin,
                          self._size - 2 * self._margin, self._size - 2 * self._margin),
                         border_radius=16)
        self._draw_routes(state)
        self._draw_orders(state)
        self._draw_depot(state)
        self._draw_hud(state, waves_total)
        return np.transpose(pygame.surfarray.array3d(self._surface), (1, 0, 2))

    def _to_screen(self, node: int, coordinates: np.ndarray) -> tuple[int, int]:
        span = self._size - 2 * self._margin
        point = coordinates[node] / _WORLD * span + self._margin
        return int(point[0]), int(point[1])

    def _draw_routes(self, state: DispatchState) -> None:
        coordinates = state.network.coordinates
        for index, vehicle in enumerate(state.vehicles.values()):
            if not vehicle.in_service or len(vehicle.route) < 2:
                continue
            points = [self._to_screen(node, coordinates) for node in vehicle.route]
            pygame.draw.lines(self._surface, _ROUTE[index % len(_ROUTE)], False, points, 3)

    def _draw_orders(self, state: DispatchState) -> None:
        assigned = state.assigned_ids()
        for order in state.orders.values():
            point = self._to_screen(order.node, state.network.coordinates)
            color = self._order_color(order, order.id in assigned)
            radius = 5 + order.demand
            pygame.draw.circle(self._surface, color, point, radius)

    def _draw_depot(self, state: DispatchState) -> None:
        x, y = self._to_screen(DEPOT, state.network.coordinates)
        pygame.draw.rect(self._surface, _DEPOT, (x - 9, y - 9, 18, 18), border_radius=4)

    def _draw_hud(self, state: DispatchState, waves_total: int) -> None:
        cost = fleet_cost(state)
        assigned = len(state.assigned_ids())
        live = len(state.live_orders())
        feasible = is_feasible(state)
        header = f"wave {state.wave}/{waves_total}   cost {cost:6.1f}   assigned {assigned}/{live}"
        self._surface.blit(self._font.render(header, True, _TEXT), (self._margin, 22))
        status = "FEASIBLE" if feasible else "INFEASIBLE"
        color = _DEPOT if feasible else _RUSH
        self._surface.blit(self._small.render(status, True, color),
                           (self._margin, self._size - 34))

    @staticmethod
    def _order_color(order, is_assigned: bool):
        if order.status is OrderStatus.CANCELLED:
            return _CANCELLED
        if order.priority is Priority.RUSH:
            return _RUSH
        return _LIVE_ASSIGNED if is_assigned else _UNASSIGNED

    def close(self) -> None:
        pygame.quit()
