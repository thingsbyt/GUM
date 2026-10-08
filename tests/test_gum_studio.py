import json
import http.client
from pathlib import Path
import threading

from gum.interface import HTML, StudioFiles, build_studio_server


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


def _request(port, method, path, *, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse(); data = response.read(); connection.close()
    return response.status, data, dict(response.getheaders())


def test_studio_requires_token_and_same_origin_for_every_mutation(tmp_path: Path):
    server, _ = build_studio_server(tmp_path / "workspace", port=0, token="test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    port = server.server_address[1]
    try:
        for path in ("/api/message", "/api/world", "/api/practice", "/api/teaching/reset"):
            status, _, _ = _request(port, "POST", path, body=b"{}",
                                     headers={"Content-Type": "application/json"})
            assert status == 403
        status, _, _ = _request(port, "POST", "/api/message", body=b'{}', headers={
            "Content-Type": "application/json", "X-GUM-Studio-Token": "test-token",
            "Host": f"127.0.0.1:{port}", "Origin": "https://attacker.example"})
        assert status == 403
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_studio_validates_json_and_serves_tokenized_page(tmp_path: Path):
    server, _ = build_studio_server(tmp_path / "workspace", port=0, token="test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    port = server.server_address[1]; auth = {"X-GUM-Studio-Token": "test-token"}
    try:
        status, page, headers = _request(port, "GET", "/?token=test-token")
        assert status == 200 and b"test-token" in page and b"__GUM_STUDIO_TOKEN__" not in page
        assert headers["X-Frame-Options"] == "DENY"
        status, _, _ = _request(port, "POST", "/api/message", body=b"{}", headers=auth)
        assert status == 400
        status, data, _ = _request(port, "POST", "/api/message",
            body=json.dumps({"text": "what do you know?"}).encode(),
            headers=auth | {"Content-Type": "application/json"})
        assert status == 200 and b"verified records" in data
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_studio_rejects_non_loopback_binding(tmp_path: Path):
    try: build_studio_server(tmp_path, host="0.0.0.0", port=0, token="test-token")
    except ValueError as error: assert "loopback" in str(error)
    else: raise AssertionError("Studio accepted a LAN-facing bind")
