from gum.dual_reveal_gate import (CONSOLE_SEEDS, CRYSTAL_SEEDS, FROZEN,
                                  RELOAD_INDICES, implementation_hashes)


def test_dual_gate_is_frozen_and_uses_twenty_new_seeds():
    assert implementation_hashes() == FROZEN
    assert len(set(CRYSTAL_SEEDS + CONSOLE_SEEDS)) == 20
    assert len(RELOAD_INDICES) == 8 and len(set(RELOAD_INDICES)) == 8
    old = set(8_101_001 + i * 4_099 for i in range(10)) | set(8_201_003 + i * 4_127 for i in range(10))
    assert not old.intersection(CRYSTAL_SEEDS + CONSOLE_SEEDS)
