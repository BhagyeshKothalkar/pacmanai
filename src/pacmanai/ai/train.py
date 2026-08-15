from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader

from pacmanai.dataset.pacman_dataset import PacmanDataset
from pacmanai.dataset import data_utils
from .model import PacmanFiLM, PacmanFiLMConfig


EPOCHS = 20
BATCH_SIZE = 8
NUM_WORKERS = 4

LEARNING_RATE = 1e-3
SCORE_SCALE = 10_000.0

DATASET_PATH = "pacman_dataset"

WANDB_PROJECT = "pacman"
WANDB_RUN_NAME = "pacman-film"

CHECKPOINT_DIR = Path("checkpoints/pacman_film")

HISTOGRAM_EVERY = 100

loss_fn = nn.MSELoss()


def tensor_stats(
    tensor: torch.Tensor,
) -> dict[str, float]:
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "abs_mean": tensor.abs().mean().item(),
        "abs_max": tensor.abs().max().item(),
    }


def model_stats(
    model: nn.Module,
) -> dict[str, float]:
    parameter_sq_sum = 0.0
    parameter_abs_sum = 0.0
    parameter_count = 0

    gradient_sq_sum = 0.0
    gradient_abs_sum = 0.0
    gradient_count = 0

    for parameter in model.parameters():
        value = parameter.detach().float()

        parameter_sq_sum += value.pow(2).sum().item()
        parameter_abs_sum += value.abs().sum().item()
        parameter_count += value.numel()

        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach().float()

        gradient_sq_sum += gradient.pow(2).sum().item()
        gradient_abs_sum += gradient.abs().sum().item()
        gradient_count += gradient.numel()

    return {
        "parameter_norm": parameter_sq_sum**0.5,
        "parameter_abs_mean": (
            parameter_abs_sum / parameter_count
        ),
        "gradient_norm": gradient_sq_sum**0.5,
        "gradient_abs_mean": (
            gradient_abs_sum / gradient_count
            if gradient_count
            else 0.0
        ),
    }


def log_model_histograms(
    model: nn.Module,
) -> None:
    for name, parameter in model.named_parameters():
        wandb.log(
            {
                f"parameters/{name}": wandb.Histogram(
                    parameter.detach().cpu().numpy()
                ),
            },
            commit=False,
        )

        if parameter.grad is not None:
            wandb.log(
                {
                    f"gradients/{name}": wandb.Histogram(
                        parameter.grad.detach().cpu().numpy()
                    ),
                },
                commit=False,
            )


def forward_model(
    model: PacmanFiLM,
    batch: dict[str, Any],
) -> torch.Tensor:
    return model(
        image=batch["image"],
        state_map=batch["state_map"],
        state_global=batch["state_global"],
        state_to_map=batch["state_to_map"],
        state_to_global=batch["state_to_global"],
        action=batch["action"],
    )


def prepare_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
    rows: int,
    cols: int,
) -> dict[str, Any]:
    return data_utils.preprocess_batch(
        batch,
        device=device,
        rows=rows,
        cols=cols,
        score_scale=SCORE_SCALE,
    )


def train_one_epoch(
    model: PacmanFiLM,
    loader: DataLoader,
    *,
    optimizer: Adam,
    device: torch.device,
    rows: int,
    cols: int,
    epoch: int,
    global_step: int,
) -> tuple[dict[str, float], int]:
    model.train()

    totals: dict[str, float] = {}

    for batch_idx, batch in enumerate(loader):
        proc_batch = prepare_batch(
            batch,
            device=device,
            rows=rows,
            cols=cols,
        )

        optimizer.zero_grad(set_to_none=True)

        pred_delta = forward_model(
            model,
            proc_batch,
        )

        target_delta = (
            proc_batch["target_image"]
            - proc_batch["image"]
        )

        loss = loss_fn(
            pred_delta,
            target_delta,
        )

        loss.backward()

        stats = model_stats(model)

        optimizer.step()

        learning_rate = optimizer.param_groups[0]["lr"]

        # -----------------------------------------------------------
        # Scalar logging.
        # -----------------------------------------------------------

        wandb_metrics: dict[str, Any] = {
            "train/loss": loss.item(),
            "train/learning_rate": learning_rate,
            "train/gradient_norm": stats["gradient_norm"],
            "train/gradient_abs_mean": stats[
                "gradient_abs_mean"
            ],
            "train/parameter_norm": stats[
                "parameter_norm"
            ],
            "train/parameter_abs_mean": stats[
                "parameter_abs_mean"
            ],
        }

        # -----------------------------------------------------------
        # Residual statistics.
        # -----------------------------------------------------------

        for name, value in tensor_stats(
            pred_delta
        ).items():
            wandb_metrics[
                f"train/pred_delta/{name}"
            ] = value

        for name, value in tensor_stats(
            target_delta
        ).items():
            wandb_metrics[
                f"train/target_delta/{name}"
            ] = value

        # -----------------------------------------------------------
        # Reconstructed image statistics.
        # -----------------------------------------------------------

        pred_image = (
            proc_batch["image"] + pred_delta
        ).clamp(0, 1)

        target_image = proc_batch["target_image"]

        for name, value in tensor_stats(
            pred_image
        ).items():
            wandb_metrics[
                f"train/pred_image/{name}"
            ] = value

        for name, value in tensor_stats(
            target_image
        ).items():
            wandb_metrics[
                f"train/target_image/{name}"
            ] = value

        wandb.log(
            wandb_metrics,
            step=global_step,
        )

        # -----------------------------------------------------------
        # Expensive histogram logging.
        # -----------------------------------------------------------

        if (
            HISTOGRAM_EVERY > 0
            and global_step % HISTOGRAM_EVERY == 0
        ):
            log_model_histograms(model)

        # -----------------------------------------------------------
        # Console.
        # -----------------------------------------------------------

        print(
            f"epoch: {epoch:03d} "
            f"batch: {batch_idx + 1:04d}/{len(loader):04d} "
            f"step: {global_step:07d} "
            f"loss: {loss.item():.6f} "
            f"grad_norm: {stats['gradient_norm']:.4f}"
        )

        # -----------------------------------------------------------
        # Epoch aggregates.
        # -----------------------------------------------------------

        epoch_metrics = {
            "loss": loss.item(),
            "gradient_norm": stats["gradient_norm"],
            "parameter_norm": stats["parameter_norm"],
        }

        for name, value in epoch_metrics.items():
            totals[name] = totals.get(name, 0.0) + value

        global_step += 1

    return (
        {
            name: value / len(loader)
            for name, value in totals.items()
        },
        global_step,
    )


@torch.no_grad()
def log_predictions(
    model: PacmanFiLM,
    batch: dict[str, Any],
    *,
    device: torch.device,
    rows: int,
    cols: int,
    epoch: int,
    global_step: int,
) -> None:
    model.eval()

    proc_batch = prepare_batch(
        batch,
        device=device,
        rows=rows,
        cols=cols,
    )

    pred_delta = forward_model(
        model,
        proc_batch,
    )

    target_delta = (
        proc_batch["target_image"]
        - proc_batch["image"]
    )

    pred_image = (
        proc_batch["image"] + pred_delta
    ).clamp(0, 1)

    target_image = proc_batch["target_image"]

    # ---------------------------------------------------------------
    # Only log a few examples.
    # ---------------------------------------------------------------

    current_image = proc_batch["image"][:4].cpu()
    pred_image = pred_image[:4].cpu()
    target_image = target_image[:4].cpu()

    pred_delta = pred_delta[:4].cpu()
    target_delta = target_delta[:4].cpu()

    # ---------------------------------------------------------------
    # Visualization transforms.
    # ---------------------------------------------------------------

    pred_delta_visual = (
        (pred_delta + 1.0) / 2.0
    ).clamp(0, 1)

    target_delta_visual = (
        (target_delta + 1.0) / 2.0
    ).clamp(0, 1)

    pred_delta_abs = pred_delta.abs().clamp(0, 1)

    delta_error = (
        pred_delta - target_delta
    ).abs().clamp(0, 1)

    # ---------------------------------------------------------------
    # Fixed-batch metrics.
    # ---------------------------------------------------------------

    delta_mse = torch.mean(
        (pred_delta - target_delta) ** 2
    )

    delta_mae = torch.mean(
        (pred_delta - target_delta).abs()
    )

    image_mse = torch.mean(
        (pred_image - target_image) ** 2
    )

    image_mae = torch.mean(
        (pred_image - target_image).abs()
    )

    wandb.log(
        {
            "eval/fixed_batch_delta_mse": delta_mse.item(),
            "eval/fixed_batch_delta_mae": delta_mae.item(),
            "eval/fixed_batch_image_mse": image_mse.item(),
            "eval/fixed_batch_image_mae": image_mae.item(),

            "images/current": [
                wandb.Image(image)
                for image in current_image
            ],

            "images/prediction": [
                wandb.Image(image)
                for image in pred_image
            ],

            "images/target": [
                wandb.Image(image)
                for image in target_image
            ],

            "images/predicted_delta": [
                wandb.Image(image)
                for image in pred_delta_visual
            ],

            "images/target_delta": [
                wandb.Image(image)
                for image in target_delta_visual
            ],

            "images/predicted_delta_abs": [
                wandb.Image(image)
                for image in pred_delta_abs
            ],

            "images/delta_error": [
                wandb.Image(image)
                for image in delta_error
            ],
        },
        step=global_step,
    )

    model.train()


def save_checkpoint(
    model: PacmanFiLM,
    optimizer: Adam,
    *,
    epoch: int,
    global_step: int,
    loss: float,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def main() -> None:
    # ---------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------

    device = torch.device("xpu")

    if not torch.xpu.is_available():
        raise RuntimeError(
            "XPU is not available."
        )

    print(
        f"Using device: {device}"
    )

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    metadata = json.loads(
        (
            Path(DATASET_PATH)
            / "metadata.json"
        ).read_text()
    )

    rows = metadata["rows"]
    cols = metadata["cols"]

    dataset = PacmanDataset(
        DATASET_PATH,
        cache_images=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    # ---------------------------------------------------------------
    # Fixed visualization batch.
    #
    # This is created once so that the same examples are visualized
    # after every epoch.
    # ---------------------------------------------------------------

    fixed_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    fixed_batch = next(iter(fixed_loader))

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    config = PacmanFiLMConfig()

    model = PacmanFiLM(
        config
    ).to(device)

    # ---------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ---------------------------------------------------------------
    # W&B
    # ---------------------------------------------------------------

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "learning_rate": LEARNING_RATE,
            "score_scale": SCORE_SCALE,
            "dataset_path": DATASET_PATH,
            "rows": rows,
            "cols": cols,
            "device": str(device),
            "model": "PacmanFiLM",
        },
    )

    # W&B model instrumentation.
    #
    # Disable graph logging because it isn't particularly useful
    # for this small model and can add overhead.
    wandb.watch(
        model,
        log="all",
        log_freq=HISTOGRAM_EVERY,
        log_graph=False,
    )

    global_step = 0

    try:
        for epoch in range(
            1,
            EPOCHS + 1,
        ):
            metrics, global_step = train_one_epoch(
                model,
                loader,
                optimizer=optimizer,
                device=device,
                rows=rows,
                cols=cols,
                epoch=epoch,
                global_step=global_step,
            )

            # -------------------------------------------------------
            # Epoch metrics.
            # -------------------------------------------------------

            wandb.log(
                {
                    "epoch/loss": metrics["loss"],
                    "epoch/gradient_norm": metrics[
                        "gradient_norm"
                    ],
                    "epoch/parameter_norm": metrics[
                        "parameter_norm"
                    ],
                    "epoch/learning_rate": optimizer.param_groups[
                        0
                    ]["lr"],
                    "epoch": epoch,
                },
                step=global_step,
            )

            # -------------------------------------------------------
            # Fixed visualizations.
            # -------------------------------------------------------

            log_predictions(
                model,
                fixed_batch,
                device=device,
                rows=rows,
                cols=cols,
                epoch=epoch,
                global_step=global_step,
            )

            # -------------------------------------------------------
            # Checkpoint.
            # -------------------------------------------------------

            save_checkpoint(
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                loss=metrics["loss"],
                path=(
                    CHECKPOINT_DIR
                    / f"pacman-film-{epoch:03d}.pt"
                ),
            )

            # Also maintain a "last" checkpoint.
            save_checkpoint(
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                loss=metrics["loss"],
                path=(
                    CHECKPOINT_DIR
                    / "last.pt"
                ),
            )

            print(
                f"epoch {epoch:03d}/{EPOCHS} "
                f"loss={metrics['loss']:.6f} "
                f"grad_norm={metrics['gradient_norm']:.4f} "
                f"param_norm={metrics['parameter_norm']:.4f}"
            )

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
