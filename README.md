# PacmanAI

PacmanAI is a FILM conditioned image edit model that learns to predict how a Pac-Man scene changes after an action. Given the current rendered frame, the game state, the next-state representation, and a movement direction, the model predicts the next frame together with the region that should be edited.
Because of the small size of the model, the game could be played locally using keyboard in real time.
The project combines a small Pygame environment, a reproducible transition-data pipeline, and a PyTorch model for conditional image prediction.

## Highlights

- Interactive Pac-Man environment rendered with Pygame
- Synthetic state-transition dataset with PNG frames and JSONL metadata
- FiLM-conditioned convolutional model with an explicit edit-mask head
- Training and evaluation metrics logged to Weights & Biases

## Requirements

- Python 3.13
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- A machine capable of running PyTorch and Pygame

## Installation

```bash
git clone https://github.com/BhagyeshKothalkar/pacmanai
cd pacmanai
uv sync
```

## Usage

### Play Pac-Man

Launch the interactive ai driven game and use the arrow keys to move. Press `R` after game over to restart.

```bash
uv run python -m pacmanai.ai.key_input
```

### Create a dataset

The generator creates random mazes and dynamic game states, applies valid random movement actions, and records the state before and after each action. It writes the dataset to the directory configured by `OUTPUT_DIR` in `src/pacmanai/dataset/gen_dataset.py`.

```bash
uv run python src/pacmanai/dataset/gen_dataset.py
```

Before generating data, adjust `NUM_EPISODES`, `EPISODES_PER_SEED`, `MAX_STEPS_PER_EPISODE`, and `OUTPUT_DIR` if needed. A generated dataset contains:

```text
<dataset>/
├── frames/             # Rendered PNG snapshots
├── metadata.json       # Dataset dimensions and schemas
├── snapshots.jsonl    # Serialized game states and frame paths
└── transitions.jsonl  # Before/after snapshot pairs and actions
```

For training and testing, configure separate dataset directories in `train.py` and `test.py` (for example, `pacman_dataset_new` and `pacman_dataset_test`).

### Train

Training uses the dataset configured by `DATASET_PATH` in `src/pacmanai/ai/train.py`. The best checkpoint is saved to `checkpoints/pacman_film/best.pt`.

```bash
uv run python src/pacmanai/ai/train.py
```

Training runs for 20 epochs by default and logs progress, metrics, and model diagnostics to Weights & Biases under the `pacman` project. Set up your W&B credentials first if remote logging is enabled in your environment.

### Evaluate

Set `TEST_DATASET_PATH` and `CHECKPOINT_PATH` in `src/pacmanai/ai/test.py`, then run:

```bash
uv run python src/pacmanai/ai/test.py
```

Evaluation reports image-reconstruction quality and edit-mask metrics, including MSE, MAE, PSNR, IoU, precision, recall, and F1.

## Model

`PacmanFiLM` is a compact convolutional encoder-decoder. It combines the current image with spatial state maps, encodes global state and the requested action, and injects that conditioning through Feature-wise Linear Modulation (FiLM). Two heads produce the next-image prediction and a sigmoid edit mask; the mask controls where the predicted residual is applied.

## Training objective

The total loss is a weighted combination of full-image reconstruction MSE, binary cross-entropy for the semantic edit mask, and MSE on pixels where the predicted and target edit regions disagree. This encourages accurate transitions while preserving pixels that should not change.

## Demo

[Watch the PacmanAI demo video](https://github.com/user-attachments/assets/e08a1431-233e-42be-b10a-63c8b5494981)

## Project structure

```text
src/pacmanai/
├── game/       # Pygame environment and maze generation
├── dataset/    # Dataset generation, loading, and preprocessing
└── ai/         # Model, training, evaluation, and input utilities
```
