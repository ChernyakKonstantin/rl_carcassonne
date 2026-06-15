import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from pycarcassone.ui import HumanGameSession
from pycarcassone.ui.server import CarcassonneUiHandler


def test_ui_server_serves_index_and_state():
    CarcassonneUiHandler.session = HumanGameSession(seed=67)
    server = ThreadingHTTPServer(("127.0.0.1", 0), CarcassonneUiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        with urlopen(f"{base_url}/", timeout=5) as response:
            html = response.read().decode("utf-8")
        with urlopen(f"{base_url}/api/state", timeout=5) as response:
            state = json.loads(response.read().decode("utf-8"))

        assert "Carcassonne" in html
        assert 'id="setup-screen"' in html
        assert 'id="current-player"' in html
        assert state["current_turn"] is not None
        assert len(state["current_turn"]["actions"]) > 0
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_ui_server_applies_action():
    CarcassonneUiHandler.session = HumanGameSession(seed=67)
    server = ThreadingHTTPServer(("127.0.0.1", 0), CarcassonneUiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        with urlopen(f"{base_url}/api/state", timeout=5) as response:
            state = json.loads(response.read().decode("utf-8"))
        action_index = state["current_turn"]["actions"][0]["index"]
        request = Request(
            f"{base_url}/api/action",
            data=json.dumps({"action_index": action_index}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            next_state = json.loads(response.read().decode("utf-8"))

        assert next_state["terminal"] or next_state["current_turn"] is not None
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_ui_server_starts_game_with_named_players():
    CarcassonneUiHandler.session = HumanGameSession(seed=67)
    server = ThreadingHTTPServer(("127.0.0.1", 0), CarcassonneUiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        request = Request(
            f"{base_url}/api/new",
            data=json.dumps(
                {
                    "seed": 67,
                    "players": [
                        {"name": "Alice", "kind": "human"},
                        {"name": "Bot", "kind": "random_bot"},
                    ],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            state = json.loads(response.read().decode("utf-8"))

        assert state["players"][0]["label"] == "Alice"
        assert state["players"][1]["kind"] == "random_bot"
        assert state["current_turn"]["player"]["label"] == "Alice"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_ui_server_starts_game_with_configured_bots():
    CarcassonneUiHandler.session = HumanGameSession(seed=67)
    server = ThreadingHTTPServer(("127.0.0.1", 0), CarcassonneUiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        request = Request(
            f"{base_url}/api/new",
            data=json.dumps(
                {
                    "seed": 67,
                    "players": [
                        {"name": "You", "kind": "human"},
                        {"name": "Bot A", "kind": "random_bot"},
                    ],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            state = json.loads(response.read().decode("utf-8"))

        assert [player["label"] for player in state["players"]] == ["You", "Bot A"]
        assert state["players"][1]["kind"] == "random_bot"
    finally:
        server.shutdown()
        thread.join(timeout=5)
