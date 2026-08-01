from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pygame

from optimbench.domain import (
    DEPOT,
    ActionType,
    Arg,
    DispatchState,
    Note,
    OrderStatus,
    Priority,
    Vehicle,
    fleet_cost,
    is_feasible,
)

_BG = (13, 19, 32)
_ARENA = (26, 34, 52)
_GRID = (34, 44, 66)
_DEPOT = (75, 212, 138)
_UNASSIGNED = (255, 182, 74)
_RUSH = (255, 92, 140)
_CANCELLED = (90, 100, 120)
_BROKEN = (120, 128, 148)
_TEXT = (214, 228, 240)
_DIM = (138, 152, 172)
# no vehicle colour may repeat a status colour: an amber route once made v1's stops
# indistinguishable from unassigned ones
_ROUTE = (
    (46, 230, 200), (86, 180, 255), (176, 148, 255),
    (222, 232, 245), (154, 240, 120), (230, 130, 255),
)
_OUTLINE = (13, 19, 32)
_WORLD = 100.0

_LEGEND = (
    (_DEPOT, "depot"),
    (_UNASSIGNED, "unassigned"),
    (_RUSH, "rush"),
    (_CANCELLED, "cancelled"),
)
_LEGEND_HINT = "assigned stops take their truck's colour"


@dataclass(frozen=True)
class FrameContext:
    """The decision that produced a frame, so a rendered episode reads as a sequence of choices.

    Rendering the state alone shows dots moving; an agent's episode is only legible when each
    frame also says which tool was called, whether the environment accepted it, and which
    disruption has just landed.
    """

    action: ActionType | None = None
    args: dict[Arg, Any] = field(default_factory=dict)
    accepted: bool = True
    note: Note = Note.NONE
    banner: str = ""
    banner_color: tuple[int, int, int] = _RUSH


_NO_CONTEXT = FrameContext()  # the initial frame, before the agent has acted


class EpisodeRenderer:
    def __init__(self, width: int = 760, height: int = 976, margin: int = 60) -> None:
        pygame.init()
        pygame.font.init()
        self._width = width
        self._height = height
        self._margin = margin
        self._arena = pygame.Rect(margin, margin + 26, width - 2 * margin, width - 2 * margin)
        self._fleet_y = self._arena.bottom + 18
        self._action_y = self._fleet_y + 74
        self._legend_y = self._action_y + 72
        self._font = pygame.font.SysFont("Menlo,Consolas,monospace", 21, bold=True)
        self._small = pygame.font.SysFont("Menlo,Consolas,monospace", 16, bold=True)
        self._tiny = pygame.font.SysFont("Menlo,Consolas,monospace", 13, bold=True)
        self._banner_font = pygame.font.SysFont("Menlo,Consolas,monospace", 25, bold=True)
        self._surface = pygame.Surface((width, height))

    def frame(self, state: DispatchState, waves_total: int,
              context: FrameContext = _NO_CONTEXT) -> np.ndarray:
        self._surface.fill(_BG)
        pygame.draw.rect(self._surface, _ARENA, self._arena, border_radius=16)
        self._draw_grid()
        self._draw_routes(state)
        self._draw_orders(state)
        self._draw_depot(state)
        self._draw_vehicles(state)
        self._draw_hud(state, waves_total)
        self._draw_fleet(state)
        self._draw_legend()
        self._draw_action_bar(context)
        if context.banner:
            self._draw_banner(context)
        return np.transpose(pygame.surfarray.array3d(self._surface), (1, 0, 2))

    def _to_screen(self, node: int, coordinates: np.ndarray) -> tuple[int, int]:
        point = coordinates[node] / _WORLD * self._arena.width
        return int(self._arena.x + point[0]), int(self._arena.y + point[1])

    def _draw_grid(self) -> None:
        for step in range(1, 5):
            offset = self._arena.width * step // 5
            pygame.draw.line(self._surface, _GRID, (self._arena.x + offset, self._arena.y),
                             (self._arena.x + offset, self._arena.bottom), 1)
            pygame.draw.line(self._surface, _GRID, (self._arena.x, self._arena.y + offset),
                             (self._arena.right, self._arena.y + offset), 1)

    def _draw_routes(self, state: DispatchState) -> None:
        coordinates = state.network.coordinates
        for index, vehicle in enumerate(state.vehicles.values()):
            if not vehicle.in_service or len(vehicle.route) < 2:
                continue
            points = [self._to_screen(node, coordinates) for node in vehicle.route]
            pygame.draw.lines(self._surface, _ROUTE[index % len(_ROUTE)], False, points, 3)

    def _draw_orders(self, state: DispatchState) -> None:
        # an assigned stop is drawn in the colour of the truck that serves it, so which truck
        # carries which order is readable without tracing a route line back to its vehicle
        carrier = self._carriers(state)
        for order in state.orders.values():
            if not 0 <= order.node < state.network.size:
                continue
            point = self._to_screen(order.node, state.network.coordinates)
            color = self._order_color(order, carrier)
            radius = 5 + order.demand
            pygame.draw.circle(self._surface, _OUTLINE, point, radius + 2)
            pygame.draw.circle(self._surface, color, point, radius)
            if order.priority is Priority.RUSH and order.status is not OrderStatus.CANCELLED:
                pygame.draw.circle(self._surface, _RUSH, point, radius + 6, 2)

    @staticmethod
    def _carriers(state: DispatchState) -> dict[str, tuple[int, int, int]]:
        return {
            order_id: _ROUTE[index % len(_ROUTE)]
            for index, vehicle in enumerate(state.vehicles.values())
            for order_id in vehicle.assigned
        }

    def _draw_depot(self, state: DispatchState) -> None:
        x, y = self._to_screen(DEPOT, state.network.coordinates)
        pygame.draw.rect(self._surface, _OUTLINE, (x - 12, y - 12, 24, 24), border_radius=6)
        pygame.draw.rect(self._surface, _DEPOT, (x - 10, y - 10, 20, 20), border_radius=5)
        self._draw_map_label("DEPOT", x, y + 15, _DEPOT)

    def _draw_map_label(self, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        # route lines run straight through these labels, so each gets a plate to sit on
        label = self._tiny.render(text, True, color)
        box = label.get_rect().inflate(8, 4)
        box.center = (x, y + label.get_height() // 2)
        pygame.draw.rect(self._surface, _BG, box, border_radius=4)
        self._surface.blit(label, (x - label.get_width() // 2, y))

    def _draw_vehicles(self, state: DispatchState) -> None:
        # only trucks that are actually out on a route are drawn on the map. Idle and broken ones
        # would stack on the depot pixel and bury it, so they live in the fleet strip instead.
        for index, vehicle in enumerate(state.vehicles.values()):
            if not vehicle.in_service or len(vehicle.route) < 3:
                continue
            node = vehicle.route[len(vehicle.route) // 2]
            x, y = self._to_screen(node, state.network.coordinates)
            color = _ROUTE[index % len(_ROUTE)]
            self._draw_truck(x, y, color)
            self._draw_map_label(self._short_id(vehicle), x, y - 27, color)

    def _draw_fleet(self, state: DispatchState) -> None:
        """One chip per vehicle: colour, id, load bar, and whether it has broken down."""
        chips = len(state.vehicles)
        gap = 10
        span = self._width - 2 * self._margin
        chip_width = (span - gap * (chips - 1)) // chips
        for index, vehicle in enumerate(state.vehicles.values()):
            x = self._margin + index * (chip_width + gap)
            alive = vehicle.in_service
            color = _ROUTE[index % len(_ROUTE)] if alive else _BROKEN
            pygame.draw.rect(self._surface, _ARENA, (x, self._fleet_y, chip_width, 58), border_radius=9)
            pygame.draw.rect(self._surface, color, (x, self._fleet_y, 4, 58),
                             border_top_left_radius=9, border_bottom_left_radius=9)
            load = vehicle.load(state.orders)
            head = self._tiny.render(self._short_id(vehicle), True, color if alive else _RUSH)
            self._surface.blit(head, (x + 14, self._fleet_y + 9))
            status = "BROKEN" if not alive else f"{load}/{vehicle.capacity}"
            text = self._tiny.render(status, True, _RUSH if not alive else _TEXT)
            self._surface.blit(text, (x + chip_width - text.get_width() - 12, self._fleet_y + 9))
            self._draw_load_bar(x + 14, self._fleet_y + 34, chip_width - 28, vehicle, load, alive, color)

    def _draw_load_bar(self, x: int, y: int, width: int, vehicle: Vehicle,
                       load: int, alive: bool, color: tuple[int, int, int]) -> None:
        pygame.draw.rect(self._surface, _GRID, (x, y, width, 8), border_radius=4)
        if not alive or load == 0:
            return
        filled = max(4, int(width * min(1.0, load / vehicle.capacity)))
        pygame.draw.rect(self._surface, color, (x, y, filled, 8), border_radius=4)

    @staticmethod
    def _short_id(vehicle: Vehicle) -> str:
        return vehicle.id.replace("veh_", "v").replace("offline", "off")

    def _draw_truck(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        pygame.draw.rect(self._surface, _OUTLINE, (x - 13, y - 10, 26, 18), border_radius=4)
        pygame.draw.rect(self._surface, color, (x - 11, y - 8, 15, 14), border_radius=3)
        pygame.draw.rect(self._surface, color, (x + 4, y - 3, 7, 9), border_radius=2)
        pygame.draw.circle(self._surface, _OUTLINE, (x - 6, y + 8), 3)
        pygame.draw.circle(self._surface, _OUTLINE, (x + 7, y + 8), 3)

    def _draw_hud(self, state: DispatchState, waves_total: int) -> None:
        cost = fleet_cost(state)
        assigned = len(state.assigned_ids())
        live = len(state.live_orders())
        header = f"wave {state.wave}/{waves_total}   cost {cost:6.1f}   assigned {assigned}/{live}"
        self._surface.blit(self._font.render(header, True, _TEXT), (self._margin, 20))
        self._draw_status_pill(is_feasible(state))

    def _draw_status_pill(self, feasible: bool) -> None:
        label, color = ("FEASIBLE", _DEPOT) if feasible else ("INFEASIBLE", _RUSH)
        text = self._small.render(label, True, _BG)
        width, height = text.get_width() + 20, text.get_height() + 10
        x = self._width - self._margin - width
        pygame.draw.rect(self._surface, color, (x, 18, width, height), border_radius=height // 2)
        self._surface.blit(text, (x + 10, 23))

    def _draw_legend(self) -> None:
        y = self._legend_y
        x = self._margin
        for color, name in _LEGEND:
            pygame.draw.circle(self._surface, color, (x + 6, y + 7), 6)
            label = self._tiny.render(name, True, _DIM)
            self._surface.blit(label, (x + 18, y))
            x += 18 + label.get_width() + 22
        hint = self._tiny.render(_LEGEND_HINT, True, _DIM)
        self._surface.blit(hint, (self._margin, y + 22))

    def _draw_action_bar(self, context: FrameContext) -> None:
        y = self._action_y
        width = self._width - 2 * self._margin
        pygame.draw.rect(self._surface, _ARENA, (self._margin, y, width, 58), border_radius=10)
        if context.action is None:
            pygame.draw.rect(self._surface, _DIM, (self._margin, y, 4, 58),
                             border_top_left_radius=10, border_bottom_left_radius=10)
            self._surface.blit(self._small.render("initial state", True, _DIM), (self._margin + 18, y + 20))
            return
        mark, mark_color = ("ACCEPTED", _DEPOT) if context.accepted else ("REJECTED", _RUSH)
        pygame.draw.rect(self._surface, mark_color, (self._margin, y, 4, 58),
                         border_top_left_radius=10, border_bottom_left_radius=10)
        call = self._small.render(self._format_call(context), True, _TEXT)
        self._surface.blit(call, (self._margin + 18, y + 10))
        badge = self._tiny.render(mark, True, mark_color)
        self._surface.blit(badge, (self._margin + 18, y + 34))
        if context.note is not Note.NONE:
            note = self._tiny.render(context.note.value, True, _DIM)
            self._surface.blit(note, (self._margin + 30 + badge.get_width(), y + 34))

    @staticmethod
    def _format_call(context: FrameContext) -> str:
        rendered = ", ".join(f"{key.value}={value}" for key, value in context.args.items())
        return f"{context.action.value}({rendered})"

    def _draw_banner(self, context: FrameContext) -> None:
        # near the top of the arena, not its centre: the depot and most routes live in the middle
        text = self._banner_font.render(context.banner, True, _BG)
        width, height = text.get_width() + 36, text.get_height() + 18
        x = (self._width - width) // 2
        y = self._arena.y + 22
        pygame.draw.rect(self._surface, context.banner_color, (x, y, width, height), border_radius=height // 2)
        self._surface.blit(text, (x + 18, y + 9))

    @staticmethod
    def _order_color(order, carrier: dict[str, tuple[int, int, int]]):
        if order.status is OrderStatus.CANCELLED:
            return _CANCELLED
        if order.id in carrier:
            return carrier[order.id]
        if order.priority is Priority.RUSH:
            return _RUSH
        return _UNASSIGNED

    def close(self) -> None:
        pygame.quit()
