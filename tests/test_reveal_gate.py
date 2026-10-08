from gum.reveal_gate import CRYSTAL_SEEDS, CONSOLE_SEEDS, FROZEN, implementation_hashes


def test_reveal_gate_is_frozen_and_seeds_are_disjoint():
    assert implementation_hashes() == FROZEN
    assert len(CRYSTAL_SEEDS) == len(CONSOLE_SEEDS) == 10
    assert len(set(CRYSTAL_SEEDS + CONSOLE_SEEDS)) == 20
