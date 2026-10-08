import unittest

from jepa_asteroids.event_options import EventOptionPlanner


class EventOptionPlannerTests(unittest.TestCase):
    def _learn_chain(self, planner):
        planner.observe("root", 1, "lit", ["event:appeared:key"], False, False)
        planner.observe("lit", 2, "open", ["event:disappeared:door"], False, False)
        planner.observations = 64; planner.coherence_streak = 32

    def test_discovers_and_composes_unrewarded_event_options(self):
        planner = EventOptionPlanner(); self._learn_chain(planner)
        planner.reset_episode()
        first = planner.recommend("root", [1, 2, 3])
        self.assertEqual(first[:2], (1, None))
        self.assertEqual(first[2], "execute-learned-event-option")
        self.assertEqual(planner.last_plan["option_count"], 2)
        planner.observe("root", 1, "lit", ["event:appeared:key"], False, False)
        second = planner.recommend("lit", [1, 2, 3])
        self.assertEqual(second[0], 2)
        planner.observe("lit", 2, "open", ["event:disappeared:door"], False, False)
        self.assertEqual(planner.status()["option_completions"], 2)

    def test_progress_credits_preceding_events(self):
        planner = EventOptionPlanner(); self._learn_chain(planner)
        planner.observe("open", 3, "goal", [], True, True)
        self.assertGreater(planner.event_progress["event:appeared:key"], 0)
        self.assertGreater(planner.event_progress["event:disappeared:door"], 0)

    def test_memory_round_trip_keeps_composable_skills(self):
        planner = EventOptionPlanner(); self._learn_chain(planner)
        memory = planner.export(); restored = EventOptionPlanner(); restored.restore(memory)
        restored.reset_episode()
        action, data, reason = restored.recommend("root", [1, 2])
        self.assertEqual((action, data, reason),
                         (1, None, "execute-learned-event-option"))
        self.assertEqual(restored.last_plan["option_count"], 2)

    def test_option_frontier_systematically_probes_controls(self):
        planner = EventOptionPlanner(); self._learn_chain(planner)
        first = planner.recommend("open", [1, 2, 3])
        self.assertEqual(first, (1, None, "probe-learned-event-frontier"))
        planner.observe("open", 1, "open", [], False, False)
        second = planner.recommend("open", [1, 2, 3])
        self.assertEqual(second[0], 2)

    def test_coordinate_payload_is_part_of_option(self):
        planner = EventOptionPlanner()
        planner.observe("root", 6, "changed", ["event:appeared:token"],
                        False, False, {"x": 17, "y": 41})
        planner.observe("changed", 1, "later", ["event:appeared:later"],
                        False, False)
        planner.observations = 64; planner.coherence_streak = 32
        planner.reset_episode()
        action, data, _ = planner.recommend("root", [6])
        self.assertEqual(action, 6)
        self.assertEqual(data, {"x": 17, "y": 41})


if __name__ == "__main__":
    unittest.main()
