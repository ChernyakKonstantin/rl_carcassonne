"""HTTP entrypoint for the browser UI.

This module intentionally stays thin: it serves static frontend assets, exposes
JSON endpoints, and delegates all game/session behavior to `HumanGameSession`.
"""

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from .session import HumanGameSession

STATIC_DIR = Path(__file__).parent / "static"


class CarcassonneUiHandler(BaseHTTPRequestHandler):
    session = HumanGameSession()

    def do_GET(self):
        if self.path == "/api/state":
            self._send_json(self.session.to_dict())
            return
        if self.path == "/" or self.path == "/index.html":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if self.path == "/app.js":
            self._send_file(STATIC_DIR / "app.js", "text/javascript; charset=utf-8")
            return
        if self.path == "/styles.css":
            self._send_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path == "/api/action":
            body = self._read_json()
            try:
                self.session.apply_action(int(body["action_index"]))
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(self.session.to_dict())
            return
        if self.path == "/api/new":
            body = self._read_json()
            seed = int(body.get("seed", 67))
            players = body.get("players")
            try:
                self.__class__.session = HumanGameSession(
                    seed=seed,
                    players=players,
                )
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(self.session.to_dict())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any):
        return

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str):
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), CarcassonneUiHandler)
    print(f"Carcassonne UI: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
