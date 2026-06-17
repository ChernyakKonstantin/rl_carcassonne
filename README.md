# Carcassonne RL Environment

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/RL-Gymnasium-green)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

A Gymnasium-compatible reinforcement learning environment for **Carcassonne**, with a Python game engine, browser UI, dynamic legal action candidates, graph-based observations, and baseline agents for self-play and benchmarking.

This project is designed as a research playground for **reinforcement learning in board games**, especially settings with:

* variable-size legal action spaces
* graph-structured game states
* self-play and multi-agent interaction
* baseline agents for reproducible comparison

## Project status

This repository is in active development.

Current focus:
* baseline/random agents for testing and benchmarking

## Why Carcassonne?

Carcassonne is a tile-placement board game with long-term planning, sparse rewards, and a naturally variable set of legal moves at each turn.

That makes it a useful environment for testing reinforcement learning agents that cannot rely on a small fixed action space.

Instead of forcing Carcassonne into a global fixed action space, this environment exposes the current legal actions directly. Each legal action can be represented as a candidate graph state and evaluated by an agent.

## Features

* Carcassonne game engine in Python
* Browser UI for human play
* Gymnasium-style RL environment
* Dynamic legal action space
* Graph-based observations
* Candidate graph generation for legal actions
* Random opponents

## Repository structure

```text
.
├── pycarcassone/
│   ├── pycarcassone/
│   │   ├── board.py          # board facade and legal tile actions
│   │   ├── graph.py          # NetworkX state graph
│   │   ├── game.py           # game loop and scoring controller
│   │   ├── player.py         # player abstractions and random player
│   │   ├── deck.py           # card deck loading and order
│   │   ├── assets/           # processed card definitions
│   │   └── ui/               # browser UI server and frontend
│   └── tests/
│
├── rl_carcassone/
│   ├── rl_carcassone/
│   │   ├── env/              # Gymnasium environment
│   │   ├── algorithm/        # RL algorithm scaffolding
│   │   └── policy/           # actor, critic, feature extractors
│   └── tests/
│
├── docs/
│   └── PROJECT_NOTES.md
│
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/ChernyakKonstantin/rl_carcassonne.git
cd rl_carcassonne
```

Create and activate a Python 3.12 environment.

Install the game engine package:

```bash
pip install -e pycarcassone
```

Install PyTorch:

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

Install PyTorch Geometric:

```bash
conda install pyg -c pyg -y
```

Install the RL package:

```bash
pip install -e rl_carcassone
```

## Running tests

From the repository root:

```bash
python -m pytest pycarcassone/tests rl_carcassone/tests -q
```

## Running the browser UI

Start the local server:

```bash
python -m pycarcassone.ui.server --host 127.0.0.1 --port 8765
```

Then open:

```text
http://127.0.0.1:8765/
```

The browser UI is an adapter over the same game engine used by the RL environment. It supports manual players, random bot players, legal tile placement selection, meeple placement, scoring updates, and visible game state.

## Reinforcement learning environment

The RL package contains a Gymnasium-style environment for training one agent against random/baseline opponents.

The environment is designed around Carcassonne's dynamic action space:

1. At each decision point, the environment computes the current legal actions.
2. Each legal action corresponds to an action index.
3. The observation includes candidate graph states for the legal actions.
4. The agent selects an integer action index from the current legal action list.
5. The environment applies that action and advances the game.

This keeps the action space close to the real structure of the game instead of flattening all possible moves into a large fixed action space.

## Training PPO

From the repository root, run the baseline PPO training config:

```powershell
& python.exe scripts/train_ppo.py --config-path config/ppo_baseline.yaml
```

For a shorter CPU-only pipeline smoke check:

```powershell
& python.exe scripts/train_ppo.py --config-path config/ppo_smoke.yaml
```

Training outputs are written under `experiments/ppo/<timestamp>/`. The baseline config uses CUDA
for PPO updates and CPU rollout workers unless the config is changed.

## Baseline agents

The project includes random agent that is useful for:

* validating the environment
* smoke-testing full episodes
* comparing trained agents against simple policies
* generating reproducible benchmark runs

## Example: environment loop

```python
from rl_carcassone.env.env import CarcassonneEnv

env = CarcassonneEnv(n_opponents=2)

obs, info = env.reset(seed=67)
terminated = False
truncated = False

while not (terminated or truncated):
    # Baseline policy: choose the first currently legal action.
    # Replace this with your policy or trained RL agent.
    action = 0

    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

## Example: random legal action

```python
import random

from rl_carcassone.env.env import CarcassonneEnv

env = CarcassonneEnv(n_opponents=2)

obs, info = env.reset(seed=67)
terminated = False
truncated = False

while not (terminated or truncated):
    n_actions = len(obs["action_candidate_graphs"])
    action = random.randrange(n_actions)

    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

## Observation design

The environment is built for action-conditioned graph evaluation.

At a decision point, the agent receives graph candidates for legal actions. A GNN-based policy can evaluate these candidate graphs and produce one logit per currently legal action.

Conceptually:

```text
current state + legal actions
        ↓
candidate graph for action 0
candidate graph for action 1
candidate graph for action 2
...
candidate graph for action N
        ↓
GNN / policy network
        ↓
N logits over current legal actions
        ↓
sample or select action
```

This design is intended for agents that operate over variable-size action sets.

## Benchmarks

Benchmark results will be added as the agent is implemented.

Planned comparisons:

| Agent               |       Opponents | Win rate | Average score | Games |
| ------------------- | --------------: | -------: | ------------: | ----: |
| Random legal action |   Random agents |      TBD |           TBD |   TBD |
| First legal action  |   Random agents |      TBD |           TBD |   TBD |
| RL agent            |   Random agents |      TBD |           TBD |   TBD |
| RL agent            | Baseline agents |      TBD |           TBD |   TBD |

## Development notes

Useful commands:

```bash
python -m pytest pycarcassone/tests -q
python -m pytest rl_carcassone/tests -q
python -m pytest pycarcassone/tests rl_carcassone/tests -q
```

Run the UI:

```bash
python -m pycarcassone.ui.server --host 127.0.0.1 --port 8765
```

## Roadmap

* [ ] Improve README examples and API documentation
* [ ] Add benchmark scripts
* [ ] Add reproducible baseline evaluation results
* [ ] Add trained policy examples
* [ ] Add more baseline agents
* [ ] Improve browser UI readability
* [ ] Add game-over/results screen
* [ ] Add model inference adapter for UI bots
* [ ] Publish package to PyPI
* [ ] Add documentation site

## Recommended GitHub topics

```text
carcassonne
carcassone
reinforcement-learning
rl-environment
gymnasium
board-game-ai
game-ai
multi-agent-reinforcement-learning
self-play
baseline-agent
graph-neural-networks
gnn
legal-action-masking
dynamic-action-space
python
```

## Citation

If you use this project in research, experiments, or a blog post, please cite it as:

```bibtex
@software{cherniak_carcassonne_rl,
  author = {Konstantin Cherniak},
  title = {Carcassonne RL Environment},
  year = {2026},
  url = {https://github.com/ChernyakKonstantin/rl_carcassonne}
}
```

## License

This repository is licensed under the Apache License 2.0.

Both packages are covered by this license:

- `pycarcassone`: Python Carcassonne engine and UI
- `rl_carcassone`: Gymnasium-compatible RL environment and baseline agents

See [LICENSE](LICENSE) for details.

## Legal notice

This is an unofficial research project for reinforcement learning experiments.

This repository is not affiliated with, endorsed by, or sponsored by Hans im Glück, Z-Man Games, Z-Man Games, Asmodee, or any other official Carcassonne rights holder.

No official Carcassonne artwork, logos, rulebook text, or proprietary assets are included in this repository.

Carcassonne is a trademark of its respective owner.
