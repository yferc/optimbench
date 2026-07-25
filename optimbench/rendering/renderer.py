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
_OUTLINE = (13, 19, 32)
_WORLD = 100.0

_LEGEND = (
    (_DEPOT, "depot"),
    (_LIVE_ASSIGNED, "assigned"),
    (_UNASSIGNED, "unassigned"),
    (_RUSH, "rush"),
    (_CANCELLED, "cancelled"),
)


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
        self._draw_legend()
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
            if not 0 <= order.node < state.network.size:
                continue
            point = self._to_screen(order.node, state.network.coordinates)
            color = self._order_color(order, order.id in assigned)
            radius = 5 + order.demand
            pygame.draw.circle(self._surface, _OUTLINE, point, radius + 1)
            pygame.draw.circle(self._surface, color, point, radius)

    def _draw_depot(self, state: DispatchState) -> None:
        x, y = self._to_screen(DEPOT, state.network.coordinates)
        pygame.draw.rect(self._surface, _OUTLINE, (x - 11, y - 11, 22, 22), border_radius=5)
        pygame.draw.rect(self._surface, _DEPOT, (x - 9, y - 9, 18, 18), border_radius=4)
        label = self._small.render("DEPOT", True, _DEPOT)
        self._surface.blit(label, (x - label.get_width() // 2, y + 14))

    def _draw_hud(self, state: DispatchState, waves_total: int) -> None:
        cost = fleet_cost(state)
        assigned = len(state.assigned_ids())
        live = len(state.live_orders())
        header = f"wave {state.wave}/{waves_total}   cost {cost:6.1f}   assigned {assigned}/{live}"
        self._surface.blit(self._font.render(header, True, _TEXT), (self._margin, 22))
        self._draw_status_pill(is_feasible(state))

    def _draw_status_pill(self, feasible: bool) -> None:
        label, color = ("FEASIBLE", _DEPOT) if feasible else ("INFEASIBLE", _RUSH)
        text = self._small.render(label, True, _BG)
        width, height = text.get_width() + 20, text.get_height() + 10
        x = self._size - self._margin - width
        pygame.draw.rect(self._surface, color, (x, 20, width, height), border_radius=height // 2)
        self._surface.blit(text, (x + 10, 25))

    def _draw_legend(self) -> None:
        y = self._size - self._margin + 20
        x = self._margin
        for color, name in _LEGEND:
            pygame.draw.circle(self._surface, color, (x + 6, y + 6), 6)
            label = self._small.render(name, True, _TEXT)
            self._surface.blit(label, (x + 18, y))
            x += 18 + label.get_width() + 22

    @staticmethod
    def _order_color(order, is_assigned: bool):
        if order.status is OrderStatus.CANCELLED:
            return _CANCELLED
        if order.priority is Priority.RUSH:
            return _RUSH
        return _LIVE_ASSIGNED if is_assigned else _UNASSIGNED

    def close(self) -> None:
        pygame.quit()
