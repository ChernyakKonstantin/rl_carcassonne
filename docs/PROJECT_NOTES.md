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

Run game + RL tests from the repository root:

```powershell
& C:/Users/cherniak/miniconda3/envs/rl_carcassone/python.exe -m pytest pycarcassone\tests rl_carcassone\tests -q
```

Run the current local PPO trainer:

```powershell
& C:/Users/cherniak/miniconda3/envs/rl_carcassone/python.exe scripts/train_ppo.py --config-path config/ppo_baseline.yaml
```

The RL package imports the sibling game package from the repository checkout during local
development. If running `rl_carcassone` as an isolated installed package, install `pycarcassone`
as well or add it to `PYTHONPATH`.

Run the human UI:

```powershell
& C:/Users/cherniak/miniconda3/envs/rl_carcassone/python.exe -m pycarcassone.ui.server --host 127.0.0.1 --port 8765
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
  Read `pycarcassone/pycarcassone/ui/README.md` for UI architecture and request flow.
- `pycarcassone/tests/`: engine, graph, and UI-session tests.
- `pycarcassone/dev_tools/`: notebooks/scripts for card data inspection/preparation.
- `rl_carcassone/rl_carcassone/env/env.py`: Gymnasium environment with graph observations and
  dynamic legal action candidates.
- `rl_carcassone/rl_carcassone/algorithm/`: RL algorithms, starting with PPO scaffolding.
- `rl_carcassone/rl_carcassone/policy/`: policy components, including actor, critic, and feature
  extractor packages.
- `rl_carcassone/rl_carcassone/data/episode.py`: raw-observation rollout containers used by PPO.
- `rl_carcassone/rl_carcassone/utils/`: local RL utilities such as config loading, explained
  variance, and single-episode rollout collection.
- `rl_carcassone/tests/`: RL environment contract tests.
- `scripts/train_ppo.py`: PPO training entry point driven by a YAML experiment config. Rollouts
  are always collected by one or more subprocess workers.
- `config/`: experiment YAML configs; `config/ppo_baseline.yaml` is the current baseline training
  config.

## Engine Contracts

- `Board.get_possible_actions(card)` returns actions that are legal for the board and current
  tile: board position, orientation, and legal meeple property choices.
- `Board` intentionally does not know current-player resources. Player-resource filtering belongs
  above the board.
- `PlayerState` is mutable player state only: game-local id, scores, and meeple resources. It does
  not imply that the player can choose actions.
- Autoplay is duck-typed: a player can be autoplayed only if its object has a callable
  `select_action(context)` method. `RandomPlayer` has one; manual UI players and RL agents do not.
  `context` is an `ActionSelectionContext` with the live board object, current player, current
  card, and legal actions. Future bot types such as ONNX players should add `select_action(context)`
  only when they can actually choose actions.
- `GameEngine.get_player_possible_actions(player, possible_actions)` filters board actions for the
  current player, including the no-meeples case.
- `GameEngine.apply_player_action(current_player, card, action)` is the shared path for applying a tile,
  optional meeple, scoring completed non-field properties, and returning meeples.
- `GameEngine` is a stateful turn-service controller for UI/RL adapters, not a standalone playable
  application. To actually play, run an adapter such as the browser UI, `CarcassonneEnv`, or another
  controller that requests turns and applies chosen actions:
  - `get_state_snapshot()` returns the adapter-facing current state: terminal flag, seed, deck
    size, dense player table, player turn order, pending turn, tile snapshot, and graph snapshot;
  - `advance_to_next_turn()` creates or returns the pending `GameTurn`;
  - `apply_turn_action(...)` / `apply_turn_action_by_index(...)` resolve the pending turn;
  - `advance_until_player_turn(player_id, autoplay=True)` autoplays other players until the target
    player must act.
- Use `advance_to_next_turn()` when an adapter wants to handle every player's decision explicitly.
  Use `advance_until_player_turn(...)` when an adapter owns one target/manual/agent player and wants
  other players with `select_action(context)` to act automatically.
- `GameEngine` assigns game-local dense player ids (`0..n-1`) in the order players are passed to the
  constructor. The engine does not assign special meaning to player `0`; adapter layers decide
  which player ids they own or train.
- `GameEngine.reset(seed=None)` resets board, deck, player resources, player turn order, and player RNG
  state. Passing a new seed changes deterministic deck/order generation.
- `Graph.locate_card_and_meeple()` assumes the action is valid. Validation should happen before
  calling it.
- `Board.resolve_outcomes(...)` is the mutating scoring API: it marks scored graph properties as
  ignored and should be used by game progression code.
- `Board.clone()` is optimized for preview branching and delegates to `Graph.clone()` instead of
  generic `deepcopy`. `Graph.clone()` copies the NetworkX structure and node/edge attribute dicts,
  including fresh `possible_values` sets for frontier positions, but does not deep-copy immutable
  card metadata.
- In `Graph._update_neighbors_possible_values()`, cached possible-neighbor sets must be copied.
  Reusing mutable `CardOption.possible_neighbors` sets corrupts global card metadata and removes
  legal placements later in a game.

## State Exports

- `GameEngine.get_state_snapshot()` is the preferred engine-level state lookup for adapters. It
  does not include action-candidate graph previews because those are expensive and should be built
  only by adapters that need them.
- `Board.get_tiles_snapshot()` / `Graph.get_tiles_snapshot()` expose placed tiles and rich property
  data for adapters:
  - tile position, values, and properties;
  - per-property type, owner, connected-component owners, ignored/scored status, and shield flag.
- The engine internally stores meeple ownership on graph property nodes. UI and RL adapters should
  prefer graph/snapshot data over inferring ownership from the raster view.
- `Board.get_graph_snapshot()` exposes graph nodes and edges for adapter-level observation builders.
  UI/RL code should not access `board._graph._graph` directly.

## Human UI Status

The browser UI is an adapter over the engine, not a rule implementation. It currently supports:

- pre-game setup screen before entering the board UI;
- creating a new game with seed and a 2-5 player setup list;
- named players where each player is either a manual human, random bot, or ONNX bot placeholder;
- manual input for every human player's turn;
- bot autoplay until the next manual player's decision;
- visible deck count, scores, and remaining meeples;
- visible current-player panel and highlighted current player row;
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

- UI reset/new-game flow currently recreates the whole `GameSession`; revisit whether it should
  reuse one session/controller and call reset/update setup explicitly.
- make tile rendering more human-readable: city areas should show diagonal boundary lines at tile
  edges where the city ends;
- improve collision handling between shield badges, meeple markers, and property labels;
- add a proper game-over/results screen;
- implement a reusable ONNX bot runtime adapter; the UI can collect an ONNX path, but ONNX bot
  setup is currently rejected until an implementation exists outside the UI adapter and can also be
  used as an RL-environment opponent;
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
preferred design. Performance matters because RL is sample-inefficient. The current implementation
keeps the candidate-graph contract but avoids the original naive clone-per-action path where
possible.

For the first implementation, `CarcassonneEnv` returns candidate graphs directly in the observation
instead of using a separate wrapper. The environment builds candidate previews from the live
`GameTurn.card`, `GameTurn.actions`, and board state, without reconstructing `CardOption` from the
encoded observation. Internally it groups actions by tile placement `(position, orientation)` and
derives meeple variants by changing only the encoded owner feature for the placed property node.
A separate candidate-graph wrapper can still be introduced later if other agent types need a
lighter base observation.

Training keeps the trainable agent in player slot `0` inside `CarcassonneEnv`. Inference adapters
for the web UI may run the same policy for any engine-assigned `player.id`, so they must build
agent-centric observations relative to the acting player: owner/player features should be encoded
as if `context.player.id` were the trainable slot `0` (or otherwise normalized to "self" versus
"opponent"). This keeps a policy trained as player `0` usable from any UI player position.

## RL Env Contract

`CarcassonneEnv` exposes one trainable agent in player slot `0` and supports up to four
`RandomPlayer` opponents, for five players total. It uses the public `GameEngine` turn-service API to
autoplay opponents until the next agent decision, similar to the human UI session adapter.

`CarcassonneEnv.reset(seed=...)` uses the project's local RNG owners instead of Gymnasium's root
`Env.np_random`: it rebuilds the `GameEngine`, `Deck`, and `RandomPlayer` opponents from the
episode seed. Do not introduce environment-level random draws unless there is a concrete need; keep
randomness local to the component that owns the stochastic behavior.

The Gym `action_space` is `DynamicDiscrete()` and validates only non-negative integer action
indices. The real legal action space is dynamic and is carried by
`observation["action_candidate_graphs"]`.
`step(a)` expects `a` to be an integer index into the current candidate graph sequence; indices
outside the current sequence are rejected.

The observation is a `spaces.Dict`:

- `action_candidate_graphs`: variable-length sequence of `HeterogeneousGraph` instances, one per
  legal action candidate. The sequence index is the action index accepted by `step`. Position,
  connector, and property nodes are separate node types with separate feature spaces; relation types
  encode position-connector links, property-connector links, connector-connector continuation,
  abbot-position links, and field-city borders. The current board graph, current tile, and raw legal
  action metadata are intentionally not separate top-level observation fields while candidate graphs
  are returned directly.
- `players`: dense player table `[player_id, score, remaining_meeples]`.
- `player_order` and `n_remaining_cards`.

Observation field notes:

- `action_candidate_graphs` is the only legal-action representation exposed to the policy. Candidate
  graph `i` semantically represents applying the live engine action `current_turn.actions[i]` to a
  board copy; `env.step(i)` applies that same action to the real game. The implementation may reuse
  placement previews across meeple/no-meeple variants as long as the encoded candidate graph is
  equivalent.
- `players` is a dense table sorted by engine player id, with rows `[player_id, score,
  remaining_meeples]`. In `CarcassonneEnv`, the trainable agent is always player `0`.
- `player_order` is the current engine turn order as player ids, preserving which players act before
  the agent's next decision after a candidate is chosen.
- `n_remaining_cards` uses `spaces.Discrete(73)`: the game has 72 cards remaining after the initial
  board tile, so valid observation values are `0..72`.
- Candidate graph node features currently use simple integer encodings:
  position nodes are `[y, x, empty]`;
  connector nodes are `[connector_type]`;
  property nodes are `[property_type, owner_id, ignored, shield, property_index]`, with `owner_id`
  set to `-1` for no owner. Edge relation types carry graph semantics; there are no per-edge feature
  values yet.

Current reward is `agent_score_delta` between agent decisions, including points scored during
opponent autoplay if they complete the agent's properties. This is a baseline contract and can be
changed deliberately when reward shaping starts.

## RL Policy And Training

The current trainable policy is an action-conditioned heterogeneous-graph actor:

- `policy/features_extractor/graph.py` converts candidate `HeterogeneousGraphInstance` observations
  to PyG `HeteroData`, adds reverse edge types, and runs `HGTConv` layers.
- Node encoders are typed: connector type, property type, and owner id use embeddings; property
  flags use numeric features; `property_index` is intentionally ignored by the extractor. Position
  nodes currently encode the `empty` flag by default and do not use absolute board coordinates
  unless `include_position_coordinates=True`.
- `Actor` returns one categorical-action logit per legal candidate graph. `Critic` pools candidate
  graph embeddings into a scalar value estimate.
- Policy constructors use the concrete environment-space contract in type annotations:
  `observation_space: gymnasium.spaces.Dict` and, for the actor, `action_space: DynamicDiscrete`.
- `AsymmetricActorCriticPolicy` is the actor/critic container with separate builders, optimizers,
  checkpoint load/save, and initialization. It currently gives actor and critic the same observation
  contract, but is kept as an extension point for a richer training-only critic observation later.

PPO is implemented locally in `rl_carcassone/rl_carcassone/algorithm/ppo/ppo.py`:

- rollouts store raw environment observations, integer action indices, log probabilities, rewards,
  values, advantages, and returns in `Episode` / `Episodes`;
- `play_single_episode(...)` collects one non-vectorized episode from `CarcassonneEnv`;
- PPO minibatches are lists of graph observations because each step has a variable-size legal
  action set;
- `to_train_actor=False` supports critic-only warmup iterations.
- `data/storage.py` persists full `Episode` objects with metadata through atomic `.tmp` to `.pt`
  renames. This is intended for worker-produced intermediate storage and future offline datasets.
- `utils/rollout_worker_pool.py` provides `RolloutWorker` and `RolloutWorkerPool`. Each
  `RolloutWorker` is the command handler running inside one subprocess: it owns its
  `CarcassonneEnv` and actor copy, receives `update_weights` commands that load the current
  `latest` actor weights, collects complete train/validation episodes, writes them to disk, and
  returns paths. `RolloutWorkerPool` is the trainer-facing facade that starts workers, broadcasts
  commands, and gathers paths before the trainer reads episodes back into RAM for PPO updates.
- Worker-pool calls are synchronous from the trainer's perspective. `update_weights(...)` blocks
  until every worker confirms the new actor weights, and `collect_episodes(...)` blocks until every
  worker has finished its assigned complete episodes and returned file paths. The trainer does not
  scan or read intermediate storage while workers are still collecting.
- Episode storage uses `.tmp` files followed by an atomic rename to `.pt`. This is a storage-level
  invariant: a visible `.pt` file is a complete serialized episode. It is not the trainer/worker
  synchronization mechanism; synchronization comes from the blocking worker responses.
- Trainer and rollout-worker devices are configured separately. The trainer builds the trainable
  actor-critic on `experiment.trainer_device`; each worker builds its rollout actor copy on
  `experiment.worker_device` and maps `latest/actor.pt` to that device when weights are updated.
  The intended baseline mode is GPU training with CPU rollout inference.

Training is configured with YAML rather than many CLI flags. `scripts/train_ppo.py` accepts only
`--config-path` and reads:

- `experiment`: output directory, trainer neural-network device, rollout-worker inference device,
  optional actor/critic checkpoint paths;
- `environment`: `CarcassonneEnv` seed and opponent count;
- `collection`: subprocess worker count (`n_workers >= 1`) plus intermediate storage directory
  name;
- `algorithm`: per-worker rollout counts, PPO hyperparameters, actor warmup period, and
  `policy_kwargs`;
- `algorithm.policy_kwargs`: actor/critic class paths, model kwargs, optimizer kwargs,
  initialization method, and `ortho_init`.

`config/ppo_baseline.yaml` is the current baseline config and uses `trainer_device: cuda` with
`worker_device: cpu`. `config/ppo_smoke.yaml` stays CPU-only for lightweight pipeline checks.
Training writes the resolved config,
flat `progress.csv` scalar metrics for table viewers, `per_epoch_critic_data.jsonl` critic
diagnostics, initial and per-iteration `latest` actor-critic checkpoints, worker-produced episodes
under `intermediate_storage/policy_<version>/{train,val}/worker_<id>`, and intermediate checkpoints
under `checkpoints/after_<N>_train_episodes` where `N` is the number of collected training
episodes.
The validation rollout branching-factor column (`val_rollout/legal_action_count_mean`) is omitted
from `progress.csv` to keep the table focused on primary training metrics.
Rollout episode counts follow stable-baselines3-style per-worker semantics:
`algorithm.n_train_episodes: 20` with `collection.n_workers: 5` collects 100 training episodes
before the PPO update.

Current RL env performance baseline after preview optimization: on seed `67`, two random
opponents, and always choosing action index `0`, a complete episode takes about 11.5 seconds on the
current development machine for 24 agent decisions and 2301 total candidate graphs. This is much
faster than the original clone-per-action `deepcopy` path, but still expensive for serious training.

## Known Risks

- Candidate graph generation is still expensive because each unique tile placement preview copies
  the board graph and encodes a full heterogeneous graph. Further training-scale optimization may
  need direct incremental graph encoding or a lighter base observation.
- PPO training still calls actor/critic on variable-size graph observations step by step, but
  rollout collection can now be parallelized through subprocess workers. Throughput is still
  limited by full candidate-graph generation and serialization cost.
- Current training is ready for smoke and pipeline experiments, but not yet for reliable
  quality-focused runs. The main unresolved experiment risks are rollout throughput, missing
  scale testing of the subprocess worker pool, no fixed evaluation protocol against random/player
  baselines, incomplete reproducibility control for torch/numpy training randomness, and no full run
  resume contract beyond loading actor/critic weights.
- Intermediate checkpoints are saved every training iteration by collected episode count. Long
  experiments may need retention or pruning rules to avoid unbounded disk growth.
- Worker intermediate episode files are large because each step stores full graph observations and
  candidate action graphs. The latest subprocess smoke run produced tens of MB per episode, so
  offline dataset accumulation will need compression, schema optimization, or cleanup policy.
- `Graph.clone()` uses NetworkX private `_node`/`_adj` structures for speed. Keep regression tests
  around preview immutability and candidate equivalence if NetworkX is upgraded.
- Observation feature normalization/encoding is intentionally simple numeric encoding. Revisit it
  before training serious GNN policies.
- UI package distribution depends on including both `pycarcassone.*` packages and
  `pycarcassone/ui/static/*` assets in `pyproject.toml`.
