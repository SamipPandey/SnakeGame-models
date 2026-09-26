
# 🐍 Snake DQN — A Deep Q-Learning Agent Written in Raw C++

A self-contained **Deep Q-Network (DQN)** that learns to play Snake from scratch — no ML frameworks, no Python, no external dependencies. Just C++ standard library, a hand-rolled neural network, Adam optimizer, and a BFS-based "safety net" planner to keep the snake from making obviously fatal moves while it's still learning.

> 🎯 **TL;DR:** Run one `g++` command, watch an agent go from bumping into walls to clearing the board, in your terminal.

---

## 📚 Table of Contents

- [How it works](#-how-it-works)
- [Project structure](#-project-structure)
- [Requirements](#-requirements)
- [Build & run](#-build--run)
- [Understanding the two source files](#-understanding-the-two-source-files)
- [Network architecture & state/action space](#-network-architecture--stateaction-space)
- [Reward shaping](#-reward-shaping)
- [The safety planner](#-the-safety-planner)
- [Using the pretrained model](#-using-the-pretrained-model)
- [Learn the theory](#-learn-the-theory-behind-this-project)
- [Roadmap ideas](#-roadmap-ideas)
- [License](#-license)

---

## 🧠 How it works

This project implements **reinforcement learning (RL)**: an agent (the snake) interacts with an environment (the board), takes actions, and receives rewards. Over many episodes it updates its own decision-making so that future reward is maximized.

There are two classic ways to represent "how good is this action in this state?" — the **Q-value**:

| Approach | How it stores Q-values | Works well when... |
|---|---|---|
| **Tabular Q-learning** | A literal table: one row per state, one column per action | The state space is small enough to enumerate (a few thousand states) |
| **Deep Q-Network (DQN)** ✅ *this project* | A neural network that takes the state as input and outputs a Q-value per action | The state space is too large/continuous for a table (ours mixes distances, directions, and flags) |

Our state representation (25 floating point features — danger flags, direction, food position, wall distances, body proximity, and safe-path info) is far too large and continuous to build a lookup table for, so this project trains a small feed-forward neural network to **approximate the Q-function** instead. That's the "Deep" in Deep Q-Learning.

Key ideas implemented here, straight from the DeepMind DQN lineage:

- ✅ **Experience Replay** — past transitions `(state, action, reward, next_state, done)` are stored in a ring buffer and sampled in random mini-batches, which breaks the correlation between consecutive frames and stabilizes training.
- ✅ **Target Network** — a slowly-updated copy of the online network is used to compute Q-learning targets, preventing the "moving target" problem.
- ✅ **Double DQN** — action *selection* uses the online network, action *evaluation* uses the target network, which reduces the classic Q-learning overestimation bias.
- ✅ **Huber loss** — a loss function that behaves like squared error near zero but like absolute error for large errors, making training more robust to outlier rewards.
- ✅ **Adam optimizer** — implemented by hand (moment estimates + bias correction), no libraries required.
- ✅ **Epsilon-greedy exploration** — the agent acts randomly with probability epsilon (decaying over time) and greedily otherwise, balancing exploration vs. exploitation.
- ✅ **BFS / flood-fill safety planner** — on top of the learned policy, a classic search algorithm double-checks that a chosen move doesn't trap the snake, so training doesn't waste episodes on avoidable, "dumb" deaths.

---

## 📁 Project structure

```
snake-dqn-rl/
├── models/
│   ├── snake_dqn_train.cpp        # Full training run (5,000 episodes, reward shaping)
│   └── snake_dqn_quicktest.cpp    # Fast smoke-test build (30 episodes) to sanity-check the build
|    
├── docs/
│   └── snake_best_model.txt # Pretrained weights — the best model found during training
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Requirements

- A C++17-capable compiler: **g++** (MinGW on Windows, or gcc/g++ on Linux/macOS) or **clang++**
- No external libraries — everything is standard C++ (`<vector>`, `<array>`, `<random>`, `<chrono>`, etc.)
- On **Windows**, the visualization step uses `<windows.h>` / `<conio.h>` for colored terminal output; on Linux/macOS these are skipped automatically via `#ifdef _WIN32`.

---

## 🚀 Build & run

Clone it, compile it, run it:

```bash
git clone https://github.com/<your-username>/snake-dqn-rl.git
cd snake-dqn-rl
```

**Full training run** (5,000 episodes, ~the "real" agent):

```bash
g++ -O3 -std=c++17 -o snake_dqn_train src/snake_dqn_train.cpp
./snake_dqn_train
```

**Quick smoke test** (30 episodes, useful to confirm your compiler setup works before committing to a long run):

```bash
g++ -O3 -std=c++17 -o snake_dqn_quicktest src/snake_dqn_quicktest.cpp
./snake_dqn_quicktest
```

> 💡 `-O3` matters a lot here — this is a hand-written matrix multiply / backprop with no SIMD libraries, so an optimized build trains meaningfully faster than a debug build.

On Windows with MinGW:

```bash
g++ -O3 -std=c++17 -o snake_dqn_train.exe src/snake_dqn_train.cpp -lgdi32
./snake_dqn_train.exe
```

While training, you'll see live progress in the terminal (episode number, score, steps, running average, survival rate, current epsilon, episodes/sec). Periodically the agent is evaluated over several games with exploration turned off, and if it beats its previous best average score, the weights are checkpointed to `snake_best_model.txt`. After training finishes (or stabilizes early), a short terminal visualization plays back the trained agent.

---

## 🔍 Understanding the two source files

Both files implement the *same* architecture — the difference is training configuration:

| Setting | `snake_dqn_train.cpp` | `snake_dqn_quicktest.cpp` |
|---|---|---|
| Episodes | 5,000 | 30 |
| Learning rate | 0.0007 | 0.0035 |
| Epsilon decay steps | 150,000 | 250,000 |
| Evaluation frequency | every 100 episodes | every 1,000 episodes |
| Reward shaping (distance-to-food bonus) | ✅ yes | ❌ no |
| Death-cause tracking (collision vs. starvation) | ✅ yes | ❌ no |

Use `snake_dqn_quicktest.cpp` to confirm everything compiles and runs on your machine in under a minute, then run the full `snake_dqn_train.cpp` for a real training session (this can take a while on CPU — it's single-threaded, dependency-free C++, not a GPU-accelerated framework).

---

## 🏗️ Network architecture & state/action space

```
Input (25 features)  →  Dense(96) + ReLU  →  Dense(96) + ReLU  →  Dense(48) + ReLU  →  Dense(3)
```

**State (25 inputs):**
- Immediate danger for going straight / turning left / turning right (3)
- Current heading direction, one-hot (4)
- Relative direction to food, up/down/left/right (4)
- Normalized distance to each wall (4)
- Nearest own-body-segment distance in each direction (4)
- Remaining safety/planner-derived features (6)

**Actions (3 — relative to the snake's current heading, not absolute directions):**
- `STRAIGHT`
- `TURN_LEFT`
- `TURN_RIGHT`

**Why relative actions instead of the usual 4 absolute directions (up/down/left/right)?** It removes the "illegal instant U-turn into your own neck" action entirely, shrinking the effective action space and making the problem easier to learn.

---

## 🎁 Reward shaping

From `snake_dqn_train.cpp`:

| Event | Reward |
|---|---|
| Eats food | `+50` |
| Survives a step, no food | `+12` shaping baseline, adjusted by distance-to-food delta, minus a small time penalty |
| Moves closer to food | small positive bonus |
| Moves away from food | small negative penalty |
| Collides with wall/self | `-10` (episode ends) |
| Times out without progress (starvation limit) | `-5` (episode ends) |

This is a fairly standard "sparse + shaped" reward setup: the big signals (eat / die) dominate, while the small distance-based shaping term helps the agent get useful gradient signal even during long stretches where no food is nearby.

---

## 🛡️ The safety planner

Pure model-free RL agents love to trap themselves — they'll happily eat toward a spot with no escape route because nothing in a single Q-value told them otherwise. `SafePlanner` adds a classical-AI layer on top of the learned policy:

- **Flood fill** from a candidate next position to estimate how much free space would remain after the move.
- **BFS reachability checks**, including a "tail is about to move, so it's usually safe to step there" heuristic.
- These signals are folded both into the **state features** (so the network can learn to prefer safe-looking states) and into **action legality checks** used during exploration, so even random exploratory moves don't waste an entire episode on an avoidable, obviously-fatal decision.

This is a common and practical pattern for RL in games: let the network learn *strategy*, and let deterministic search handle *obviously bad moves*, so training time isn't spent re-learning "don't trap yourself" from scratch every run.

---

## 🎮 Using the pretrained model

`models/snake_best_model.txt` is a plain-text dump of the network's weights and biases, layer by layer (`inSize outSize` header followed by the flattened weight matrix and bias vector for each layer). To use it, point the loader at this file — see the `save()` / `load()` methods in `src/snake_dqn_train.cpp` for the exact format if you want to load it into a visualization-only build, or use it as a warm start for further training.

---

## 📖 Learn the theory behind this project

If Q-learning, DQN, or any of the above is new to you, these are the resources that explain the ideas this code implements:

- 📘 **Reinforcement Learning: An Introduction** (Sutton & Barto) — the canonical RL textbook, free PDF: http://incompleteideas.net/book/the-book-2nd.html
- 📄 **Playing Atari with Deep Reinforcement Learning** (Mnih et al., DeepMind's original DQN paper): https://arxiv.org/abs/1312.5602
- 📄 **Human-level control through deep reinforcement learning** (DQN with target networks, published in *Nature*): https://www.nature.com/articles/nature14236
- 📄 **Deep Reinforcement Learning with Double Q-learning** (the Double DQN fix used in this project): https://arxiv.org/abs/1509.06461
- 🎓 **Hugging Face Deep RL Course** (free, hands-on): https://huggingface.co/learn/deep-rl-course
- 🎓 **OpenAI Spinning Up in Deep RL** (concept explainers + code): https://spinningup.openai.com/
- 🎥 **David Silver's RL Course (DeepMind/UCL)** on YouTube — the classic video lecture series on Q-learning, value functions, and policy gradients.

---

## 🗺️ Roadmap ideas

Contributions and forks welcome — some natural next steps:

- [ ] Multi-threaded / SIMD batch training for faster convergence
- [ ] Prioritized experience replay
- [ ] Larger board sizes / obstacles
- [ ] A lightweight SDL or terminal-graphics visualizer instead of ANSI escape codes
- [ ] Export weights to ONNX so they can be loaded from Python/PyTorch too

If this project helped you understand DQN, **consider giving it a ⭐** — it helps other people learning RL find it.

---

## 📄 License
---
MIT — see [LICENSE](LICENSE). Do whatever you want with it, attribution appreciated.
