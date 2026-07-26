import pytest

from optimbench.domain import Difficulty, OrderStatus
from optimbench.generation import DIFFICULTY, DispatchScenarioGenerator

GEN = DispatchScenarioGenerator()


@pytest.mark.parametrize("difficulty", list(Difficulty))
@pytest.mark.parametrize("seed", range(6))
def test_capacity_slack_makes_demand_satisfiable(difficulty, seed):
    scenario = GEN.generate(seed, difficulty)
    state = scenario.state
    capacity = sum(v.capacity for v in state.vehicles.values() if v.in_service)
    demand = sum(o.demand for o in state.live_orders())
    assert capacity >= demand


@pytest.mark.parametrize("difficulty", list(Difficulty))
def test_vehicles_start_cleared(difficulty):
    state = GEN.generate(0, difficulty).state
    assert all(not v.assigned and not v.route for v in state.vehicles.values())


def test_disruption_count_matches_spec():
    assert len(GEN.generate(0, Difficulty.EASY).disruptions) == DIFFICULTY[Difficulty.EASY].disruption_waves
    assert len(GEN.generate(0, Difficulty.HARD).disruptions) == DIFFICULTY[Difficulty.HARD].disruption_waves


def test_determinism_same_seed():
    a = GEN.generate(7, Difficulty.MEDIUM)
    b = GEN.generate(7, Difficulty.MEDIUM)
    assert list(a.state.orders) == list(b.state.orders)
    assert [d.type for d in a.disruptions] == [d.type for d in b.disruptions]


def test_cancelled_distractors_present_on_hard():
    cancelled = [o for o in GEN.generate(0, Difficulty.HARD).state.orders.values()
                 if o.status is OrderStatus.CANCELLED]
    assert cancelled
