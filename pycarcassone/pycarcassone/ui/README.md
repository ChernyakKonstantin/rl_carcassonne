# Human UI

This package is the browser adapter for the `pycarcassone` engine.
It is not a separate rules implementation. Rules, legal actions, turn order,
scoring, deck state, and player resources live in `GameEngine`, `Board`, and `Graph`.

The UI package provides three layers:

- `server.py`: a small stdlib HTTP server. It serves static files and exposes
  JSON endpoints.
- `session.py`: the adapter between HTTP/UI state and the engine turn-service
  API.
- `static/`: browser-only rendering and interaction code.

## How To Run

From the repository root:

```powershell
& C:/Users/cherniak/miniconda3/envs/rl_carcassone/python.exe -m pycarcassone.ui.server --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765/`.

## Runtime Shape

`CarcassonneUiHandler` keeps one process-local `HumanGameSession` in a class
attribute. All browser requests talk to that session.

Current limitation: starting a new game recreates the whole `GameSession`.
This is deliberate for now, but should be revisited if the UI needs a clearer
reset/update lifecycle.

## Request Flow

The browser never mutates the engine directly. It talks to HTTP endpoints:

- `GET /`: serves `static/index.html`.
- `GET /app.js`: serves browser logic.
- `GET /styles.css`: serves styling.
- `GET /api/state`: returns the serialized session state.
- `POST /api/new`: creates a new session from seed and player setup.
- `POST /api/action`: applies a selected action index for the current manual
  turn.

The frontend receives legal action indices from `/api/state` and sends one of
those indices back to `/api/action`. It does not recompute Carcassonne legality.

## Session Responsibilities

`GameSession` owns UI-specific concerns:

- normalize player setup from JSON-like dictionaries;
- build engine player objects (`PlayerState` for manual players, `RandomPlayer` for random bots);
- remember player display metadata: names, colors, kind, manual/bot flag;
- advance the engine until a manual player must act;
- autoplay bot turns for players that implement `select_action(context)`;
- serialize engine state into JSON-friendly structures for the browser.

`GameSession` does not duplicate pending turn state. The source of truth is
`GameEngine.current_turn`.

Autoplay context passes the live board object plus current player, current card,
and legal actions. Bot implementations choose their own observation form from
that context, for example graph snapshot, tile snapshot, raster view, or a
custom JSON-like structure.

## Player Kinds

Supported setup values:

- `human`: manual player controlled through the browser.
- `random_bot`: bot using `RandomPlayer`.
- `onnx_bot`: placeholder. The UI can collect a model path, but session
  creation currently rejects this kind until a reusable ONNX bot implementation exists outside the
  UI adapter.

At least one `human` player is required, otherwise there is no browser-controlled
turn to stop on.

## Frontend State

`static/app.js` keeps only transient browser selection:

- selected tile orientation;
- selected board position;
- selected meeple property or no-meeple choice.

Durable game state comes from `/api/state`.

The board is rendered from `Board.get_tiles_snapshot()` data serialized by the
session. Current-card previews and legal placement targets are rendered from
`GameEngine.current_turn.card` and `GameEngine.current_turn.actions`.

## Known UI Backlog

- Implement a reusable ONNX bot runtime adapter that can be used by both UI sessions and RL envs.
- Replace session recreation on new game with an explicit reset/update flow if
  needed.
- Add a proper game-over/results screen.
- Improve collision handling between property labels, meeples, and city shields.
- Decide whether large boards need zoom, pan, or fit-to-content.
