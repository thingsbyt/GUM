import json
from pathlib import Path

from gum.interface import HTML, StudioFiles


def test_studio_contains_live_teaching_evidence_brain_and_document_views():
    for text in ("Live lab", "Teaching lab", "Evidence", "Brain & lineage", "Documentation",
                 "Nothing trains automatically", "Teach by showing", "Run fresh 45-trial audit"):
        assert text in HTML


def test_studio_serves_only_files_declared_inside_release(tmp_path: Path):
    media = tmp_path / "media"; media.mkdir(); replay = media / "replay.gif"; replay.write_bytes(b"GIF89a")
    docs = tmp_path / "docs"; docs.mkdir(); paper = docs / "paper.md"; paper.write_text("paper", encoding="utf-8")
    outside = tmp_path.parent / "outside.txt"; outside.write_text("not public", encoding="utf-8")
    index = {
        "claims": [],
        "replays": [{"id": "replay", "file": "media/replay.gif"}],
        "documents": [
            {"id": "paper", "file": "docs/paper.md"},
            {"id": "escape", "file": "../outside.txt"},
        ],
    }
    (tmp_path / "EVIDENCE_INDEX.json").write_text(json.dumps(index), encoding="utf-8")
    studio = StudioFiles(tmp_path)
    assert studio.media("replay") == replay.resolve()
    assert studio.document("paper") == paper.resolve()
    assert studio.document("escape") is None
    assert studio.media("unknown") is None
