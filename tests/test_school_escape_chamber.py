"""Physics, boundary, and rendering contracts for cooperative Room A."""

import numpy as np
import pytest

from gum.school.escape_chamber import (
    DEVELOPMENT_ADAPTER,
    EAST,
    NORTH,
    SOUTH,
    WAIT,
    WEST,
    EscapeChamberError,
    EscapeChamberDevelopment,
    EscapeChamberRoomA,
    RewardTable,
    room_a_invariant_report,
)


def _joint(world, actions):
    wait = world.action_slot_for_test(WAIT)
    return [
        None if world.escaped[index] else world.action_slot_for_test(actions.get(index, WAIT))
        for index in range(4)
    ]


def _move(world, member, semantic):
    return world.step(_joint(world, {member: semantic}))


def _positive_control(world):
    for action in (WEST, SOUTH, SOUTH):
        _move(world, 0, action)
    assert world.positions[0] == world.plate
    routes = (
        (EAST, EAST, EAST, SOUTH, SOUTH, EAST, EAST, EAST, EAST, EAST),
        (NORTH, EAST, EAST, EAST, EAST, NORTH, EAST, EAST, EAST, EAST, EAST),
        (EAST, EAST, EAST, NORTH, NORTH, EAST, EAST, EAST, EAST, EAST),
    )
    final = None
    for member, route in zip((1, 2, 3), routes, strict=True):
        for action in route:
            final = _move(world, member, action)
    return final


def test_public_contract_and_member_views_are_bounded_pixels():
    world = EscapeChamberRoomA(seed=101)
    observations = world.reset()
    spec = world.public_spec()
    assert spec.agents == 4 and spec.action_count == 5 and spec.protocol_version == 1
    assert spec.observation_shape == (96, 160, 3)
    assert len(observations) == 4
    assert all(frame.shape == spec.observation_shape and frame.dtype == np.uint8
               for frame in observations)
    assert len({frame.tobytes() for frame in observations}) == 4
    assert world.spectator_frame().shape == (720, 1280, 3)
    transition = world.step(_joint(world, {})).transition_for(0)
    assert set(transition.public_info) == {"contract_version", "episode_boundary"}


def test_shared_anonymous_action_map_is_reproducible_and_changes_between_seeds():
    left = EscapeChamberRoomA(seed=201); left.reset()
    same = EscapeChamberRoomA(seed=201); same.reset()
    other = EscapeChamberRoomA(seed=202); other.reset()
    assert left.audit_state()["action_map"] == same.audit_state()["action_map"]
    assert left.audit_state()["action_map"] != other.audit_state()["action_map"]
    assert sorted(left.audit_state()["action_map"]) == ["east", "north", "south", "wait", "west"]


def test_single_body_cannot_reach_gate_outside_under_exhaustive_position_search():
    report = room_a_invariant_report()
    assert report["single_body_escape_possible"] is False
    assert report["single_body_reachable_cells"] == 49
    assert report["plate_gate_manhattan_distance"] >= 5


def test_development_room_simplifies_geometry_without_supplying_holder():
    world = EscapeChamberDevelopment(seed=881, horizon=3)
    world.reset()

    assert world.public_spec().adapter == DEVELOPMENT_ADAPTER
    assert world.plate not in world.positions
    assert world.gate_inside in world.positions
    assert abs(world.plate[0] - world.gate_inside[0]) == 2

    step = world.step(_joint(world, {0: EAST, 1: EAST}))

    assert step.resolution["gate_open"] is True
    assert step.resolution["gate_crossers"] == [1]


def test_isolated_positive_control_gets_exactly_three_out_and_keeps_holder_inside():
    world = EscapeChamberRoomA(seed=303, horizon=100)
    world.reset()
    final = _positive_control(world)
    assert final is not None and final.terminated and not final.truncated
    assert final.escaped_this_tick == (3,)
    assert world.escape_order == [1, 2, 3]
    assert world.positions[0] == world.plate
    assert sum(world.escaped) == 3
    assert world.gate_open is True
    with pytest.raises(EscapeChamberError, match="episode is over"):
        world.step([0, None, None, None])


def test_holder_departure_closes_gate_before_same_tick_crossing():
    world = EscapeChamberRoomA(seed=404)
    world.reset()
    world.positions = [world.plate, world.gate_inside, (3, 6), (4, 6)]
    result = world.step(_joint(world, {0: EAST, 1: EAST}))
    assert world.positions[0] == (3, 4)
    assert world.positions[1] == world.gate_inside
    assert result.resolution["gate_open"] is False
    assert result.resolution["gate_crossers"] == []


def test_replacement_holder_can_open_gate_on_same_tick():
    world = EscapeChamberRoomA(seed=405)
    world.reset()
    world.positions = [world.plate, world.gate_inside, (2, 3), (4, 6)]
    result = world.step(_joint(world, {0: EAST, 1: EAST, 2: SOUTH}))
    assert world.positions[0] == (3, 4)
    assert world.positions[2] == world.plate
    assert world.positions[1] == world.gate_outside
    assert result.resolution["gate_open"] is True
    assert result.resolution["gate_crossers"] == [1]


def test_collision_and_swap_rules_have_no_identity_priority():
    world = EscapeChamberRoomA(seed=505)
    world.reset()
    world.positions = [(3, 4), (5, 4), (3, 6), (4, 6)]
    result = world.step(_joint(world, {0: EAST, 1: WEST}))
    assert world.positions[:2] == [(3, 4), (5, 4)]
    assert result.resolution["ordinary_cancelled"] == [0, 1]
    world.positions = [(3, 4), (4, 4), (3, 6), (4, 6)]
    result = world.step(_joint(world, {0: EAST, 1: WEST}))
    assert world.positions[:2] == [(3, 4), (4, 4)]
    assert result.resolution["ordinary_cancelled"] == [0, 1]


@pytest.mark.parametrize("seed", range(12))
def test_adversarial_random_schedules_never_allow_four_escapes(seed):
    world = EscapeChamberRoomA(seed=600 + seed, horizon=180)
    world.reset()
    rng = np.random.default_rng(70_000 + seed)
    while not world._done:
        actions = [None if escaped else int(rng.integers(5)) for escaped in world.escaped]
        world.step(actions)
        assert sum(world.escaped) <= 3
    assert sum(world.escaped) <= 3


def test_team_individual_and_mixed_rewards_match_precommitted_table():
    results = {}
    for treatment in ("team", "individual", "mixed"):
        world = EscapeChamberRoomA(
            seed=707,
            reward_table=RewardTable(treatment=treatment),
        )
        world.reset()
        world.positions = [world.plate, world.exit_inside, (3, 6), (4, 6)]
        step = world.step(_joint(world, {1: EAST}))
        results[treatment] = step.rewards
    assert results["team"] == pytest.approx((0.999, 0.999, 0.999, 0.999))
    assert results["individual"] == pytest.approx((-0.001, 0.999, -0.001, -0.001))
    assert results["mixed"] == pytest.approx((0.499, 0.999, 0.499, 0.499))


def test_escaped_body_is_inactive_cannot_return_and_still_receives_team_credit():
    world = EscapeChamberRoomA(seed=808, reward_table=RewardTable(treatment="team"))
    world.reset()
    world.positions = [world.plate, world.exit_inside, (3, 6), (4, 6)]
    world.step(_joint(world, {1: EAST}))
    assert world.escaped[1] and world.positions[1] is None
    with pytest.raises(EscapeChamberError, match="emit no action"):
        world.step([world.action_slot_for_test(WAIT)] * 4)
    world.positions[2] = world.exit_inside
    step = world.step(_joint(world, {2: EAST}))
    assert step.rewards[1] == pytest.approx(1.0)


def test_horizon_truncates_and_replay_is_deterministic_from_seed_and_actions():
    actions = []
    left = EscapeChamberRoomA(seed=909, horizon=8); left.reset()
    rng = np.random.default_rng(10)
    while not left._done:
        joint = [int(rng.integers(5)) for _ in range(4)]
        actions.append(joint)
        final_left = left.step(joint)
    right = EscapeChamberRoomA(seed=909, horizon=8); right.reset()
    for joint in actions:
        final_right = right.step(joint)
    assert final_left.truncated and final_right.truncated
    assert left.audit_state()["state_sha256"] == right.audit_state()["state_sha256"]
    assert np.array_equal(left.spectator_frame(), right.spectator_frame())
