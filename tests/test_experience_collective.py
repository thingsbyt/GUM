from gum.experience_collective import ExperienceCollective


def test_collective_exchanges_world_memory_without_aliasing():
    team = ExperienceCollective(); core = team.members[0].shared.core
    core.worlds["world-a"] = {"value": [1]}; core.form_roles[7]["collectible"] = 3
    team.publish(0)
    other = team.members[1].shared.core
    assert other.worlds["world-a"] == {"value": [1]}
    assert other.form_roles[7]["collectible"] == 3
    other.worlds["world-a"]["value"].append(2)
    assert core.worlds["world-a"]["value"] == [1]
