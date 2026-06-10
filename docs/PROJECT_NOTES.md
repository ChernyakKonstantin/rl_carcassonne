# Project Notes

These notes are a quick orientation layer for future Codex chats and for the project owner.
Read `AGENTS.md` first, then this file.

This file is intended to be a curated, versioned project map. Keep durable architecture,
contracts, and project direction here. Put volatile chat notes, temporary status, and scratch
decisions in `docs/WORKING_NOTES.md`; that file is local and gitignored.

## Current Goal

The project is a Carcassonne implementation intended to become the game backend for
Gymnasium-based RL experiments. The planned RL direction is GNN-based, so dynamic graph-shaped
observations are an intentional design direction, not a problem by itself.

The nearest practical goal is a human-play interface on top of the existing engine. The human
player needs enough game-state visibility to make legal Carcassonne decisions, while the other
players can remain arbitrary `BasePlayer` subclasses.

For the human interface, the current missing pieces are:

- Clear visualization of which player owns meeples on which properties.
- Clear display of the current tile.
- Clear display of all legal placements for the current tile.
- Ability to rotate the current tile.
- Ability to place no meeple or place a meeple on a selected legal property.
- Visible meeple counters for the human player and opponents.
- Visible number of remaining deck cards.

The human interface should not force an architecture that makes RL integration harder. Prefer
engine APIs that expose state and legal decisions cleanly, then build both UI and RL adapters on
top of those APIs.

## Repository Map

- `pycarcassone/` contains the game package.
- `pycarcassone/pycarcassone/` contains the actual engine code.
- `pycarcassone/pycarcassone/assets/cards.json` contains processed tile definitions.
- `pycarcassone/pycarcassone/assets/cards_raw.json` contains source/raw tile data.
- `pycarcassone/tests/` contains game/graph tests.
- `pycarcassone/dev_tools/` contains notebooks used to prepare/inspect card data.
- `rl_carcassone/` is reserved for the Gymnasium environment, RL agents, neural networks,
  training loops, and related code.
- `rl_carcassone/env/env.py` is currently a commented-out Gymnasium env stub.

## Python Environment

Use the `rl_carcassone` conda environment. In this workspace the interpreter path is:

```powershell
& C:/Users/cherniak/miniconda3/envs/rl_carcassone/python.exe
```

Example test command:

```powershell
& C:/Users/cherniak/miniconda3/envs/rl_carcassone/python.exe -m pytest pycarcassone\tests -q
```

As of 2026-06-10, this command passed: `4 passed`. Pytest may warn that it cannot write cache
files under some directories because of local permissions.

## Core Engine Responsibilities

- `Game` owns the deck, player order, board, turn loop, scoring application, and rendering entrypoint.
- `Board` is the public board-level facade. It places cards, computes outcomes, and exposes
  board-local possible actions.
- `Graph` is the detailed state model. It stores positions, connectors, properties, owners,
  property links, field-city borders, completion checks, and scoring helpers.
- `Deck` loads card definitions and shuffles remaining cards.
- `Card` and `CardOption` represent tile variants by orientation.
- `BasePlayer` and its subclasses select actions and own player resources such as score and
  remaining meeples.

## Turn Flow

Current `Game._loop_step()` flow:

1. Rotate to the next current player.
2. Draw the next deck card that has at least one board-local legal action.
3. Ask the player to select one action from the possible actions.
4. If the selected action places a meeple, decrement the player's remaining meeples.
5. Place the card and optional meeple through `Board.put_card_and_meeple()`.
6. Score completed non-field properties and return meeples.

At game end, `Game._process_outcomes()` scores remaining properties and fields.

## Important Contracts

- `Board.get_possible_actions(card)` returns actions legal with respect to the board and the
  current card: tile position, orientation, and allowed meeple property positions.
- `Board` does not know the current player's resource state. Actions that require a meeple are
  player-legal only if the current player has remaining meeples.
- Existing `RandomPlayer` handles the no-meeples case by filtering out actions with
  `meeple_position is not None`.
- Future Gymnasium action masks should be built at the `Game`/`Env` layer, where both
  `possible_actions` and `current_player.remaining_meeples` are available.
- `Graph.locate_card_and_meeple()` assumes its inputs are valid and does not perform full
  validation.
- `Board.get_outcomes()` is not a pure query: it mutates graph state by marking scored properties
  as ignored.

## Observation Notes

- `Board.get_view()` currently returns a raster-like numpy array of `PixelMeaning` values.
- `Board.get_view()` still does not include meeple positions/owners.
- Meeple ownership is not lost internally: `Graph` stores `owner` on property nodes when a meeple
  is placed.
- For the planned GNN-based RL agent, prefer a graph observation adapter over extending
  `Board.get_view()` as the main RL observation.
- A useful future graph observation should expose node features, edge indices/features, current
  tile information, player scores, remaining meeples, current player id, and a legal action list
  or mask.

## RL Direction

The first RL concept to try is action-conditioned graph evaluation:

1. At a decision point, the environment has the current board state and `N` legal ways to place the
   current tile: position, orientation, and no meeple or a meeple on a specific property.
2. Build `N` candidate graph states by cloning the current graph and applying one legal action to
   each clone.
3. Send those `N` candidate graphs as a batch through a GNN.
4. The GNN returns `N` logits, one per legal candidate action.
5. Apply softmax over the variable-size candidate set and sample/select the action.

This keeps the action space variable-sized, which is natural for Carcassonne. A fixed action space
is not the preferred design direction for this project.

Performance matters because RL is sample-inefficient. Before committing to the clone-per-action
approach, benchmark graph cloning and candidate generation. If NetworkX graph cloning becomes a
bottleneck, consider a faster candidate-state representation, structural sharing, incremental
features, or scoring action candidates without fully materializing every successor graph.

## Known Issues / Review Notes

- `BasePlayer.select_action()` signature says `(current_board_state, current_card, possible_actions)`,
  but `Game._loop_step()` currently calls it as `(card, board_view, possible_actions)`.
- `Game.reset()` resets deck and board, but does not currently reset player scores or remaining
  meeples.
- `Game.rng` is created without a seed even though `Deck` is seeded.
- `BasePlayer.id` uses `uuid4().int`; stable dense player ids may be more convenient for RL.
- `Game.PLAYER_ID = 0` does not currently line up with `BasePlayer.id` generated by UUID.
- `Board.get_view()` has dynamic spatial shape. This is acceptable for the planned GNN direction,
  but Gymnasium integration still needs a clear observation space contract.

## Git History Notes

- Commit `298aa9d0cdef227997dc0828ac65307ca325b39b` added field scoring support, not meeple
  observation export.
- Commit `433a42c` separated the game package from the RL package, moving the game from
  `rl_carcassone/env/game` to `pycarcassone/pycarcassone` with 100% similarity for the moved files.
