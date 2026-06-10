# Project Notes

Quick lookup for future Codex chats. This is not a changelog: keep current project structure,
contracts, decisions, and active risks here. Read `AGENTS.md` first. If local
`docs/WORKING_NOTES.md` exists, read it after this file for temporary scratch status.

## Project Shape

The project has two related layers:

- `pycarcassone/`: Carcassonne game engine plus a browser UI for human play.
- `rl_carcassone/`: future Gymnasium environment, RL agents, neural networks, training code, and
  related experiments.

The immediate product goal is a usable human-play UI on top of the engine. The strategic goal is
to reuse the same engine contracts for RL, especially a GNN-based agent with a variable-size legal
action set.

## Environment

Use the `rl_carcassone` conda environment directly by interpreter path:

```powershell
& C:/Users/cherniak/miniconda3/envs/rl_carcassone/python.exe
```

Run tests:

```powershell
& C:/Users/cherniak/miniconda3/envs/rl_carcassone/python.exe -m pytest pycarcassone\tests -q
```

Run the human UI:

```powershell
& C:/Users/cherniak/miniconda3/envs/rl_carcassone/python.exe -m pycarcassone.pycarcassone.ui.server --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765/`.

## Repository Map

- `pycarcassone/pycarcassone/board.py`: board facade and board-local legal actions.
- `pycarcassone/pycarcassone/graph.py`: detailed NetworkX state model for tiles, connectors,
  properties, owners, completion, and scoring helpers.
- `pycarcassone/pycarcassone/game.py`: deck/player turn loop, current-player legality, scoring
  application, game reset/end handling.
- `pycarcassone/pycarcassone/player.py`: player abstractions, resources, and action selection.
- `pycarcassone/pycarcassone/deck.py`: card loading and deck order.
- `pycarcassone/pycarcassone/assets/cards.json`: processed card definitions.
- `pycarcassone/pycarcassone/assets/cards_raw.json`: raw/source card definitions.
- `pycarcassone/pycarcassone/ui/`: stdlib HTTP server, human-game session adapter, and static UI.
- `pycarcassone/tests/`: engine, graph, and UI-session tests.
- `pycarcassone/dev_tools/`: notebooks/scripts for card data inspection/preparation.
- `rl_carcassone/env/env.py`: currently a commented-out Gymnasium env stub.

## Engine Contracts

- `Board.get_possible_actions(card)` returns actions that are legal for the board and current
  tile: board position, orientation, and legal meeple property choices.
- `Board` intentionally does not know current-player resources. Player-resource filtering belongs
  above the board.
- `Game.get_player_possible_actions(player, possible_actions)` filters board actions for the
  current player, including the no-meeples case.
- `Game.apply_player_action(current_player, card, action)` is the shared path for applying a tile,
  optional meeple, scoring completed non-field properties, and returning meeples.
- `Game` assigns game-local dense player ids (`0..n-1`) in the order players are passed to the
  constructor. `Game.PLAYER_ID == 0` is the main agent/human slot by convention.
- `Game.reset(seed=None)` resets board, deck, player resources, player turn order, and player RNG
  state. Passing a new seed changes deterministic deck/order generation.
- `Graph.locate_card_and_meeple()` assumes the action is valid. Validation should happen before
  calling it.
- `Board.resolve_outcomes(...)` is the mutating scoring API: it marks scored graph properties as
  ignored and should be used by game progression code.
- In `Graph._update_neighbors_possible_values()`, cached possible-neighbor sets must be copied.
  Reusing mutable `CardOption.possible_neighbors` sets corrupts global card metadata and removes
  legal placements later in a game.

## State Exports

- `Board.get_view()` returns the older raster-like numpy view of `PixelMeaning` values. It is useful
  for simple visualization but is not the main intended RL observation.
- `Board.get_tiles_snapshot()` / `Graph.get_tiles_snapshot()` expose placed tiles and rich property
  data for adapters:
  - tile position, values, and properties;
  - per-property type, owner, connected-component owners, ignored/scored status, and shield flag.
- The engine internally stores meeple ownership on graph property nodes. UI and RL adapters should
  prefer graph/snapshot data over inferring ownership from the raster view.

## Human UI Status

The browser UI is an adapter over the engine, not a rule implementation. It currently supports:

- creating a new game with seed and opponent count;
- human player plus `RandomPlayer` opponents;
- opponent autoplay until the next human decision;
- visible deck count, scores, and remaining meeples;
- board rendering from tile snapshots;
- current tile previews for every available orientation;
- selecting orientation by clicking a tile preview;
- dashed legal-placement targets for the selected orientation;
- selecting no meeple or a legal meeple property after choosing placement;
- meeple markers on properties;
- city shield badges on shielded city properties;
- striped ownership overlays for field/road/city components using connected-component owners.

Current automated UI coverage is session-level / in-process HTTP coverage. No browser automation
stack is configured.

UI backlog / open questions:

- improve collision handling between shield badges, meeple markers, and property labels;
- add a proper game-over/results screen;
- decide whether large boards need zoom, pan, or fit-to-content.

Rejected or deprioritized UI ideas for now:

- ghost preview of the current tile directly inside legal placements;
- extra property highlighting before meeple selection.

## RL Direction

The preferred first RL design is action-conditioned graph evaluation:

1. At a decision point, get the current board state and `N` legal actions for the current tile.
2. Build `N` candidate graph states by applying one legal action per candidate.
3. Batch those candidate graphs through a GNN.
4. Produce `N` logits, softmax over the variable-size action set, then sample/select an action.

This keeps Carcassonne's naturally variable action space. A fixed global action space is not the
preferred design. Performance matters because RL is sample-inefficient; benchmark graph cloning
and candidate generation before committing to a clone-per-action implementation.

## Known Risks

- `Board.get_view()` has dynamic spatial shape. That is acceptable for the GNN direction, but the
  Gymnasium env still needs an explicit observation/action contract.
- UI package distribution depends on including both `pycarcassone.*` packages and
  `pycarcassone/ui/static/*` assets in `pyproject.toml`.
