from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader

from pacmanai.dataset.pacman_dataset import PacmanDataset
from pacmanai.dataset import data_utils
from .model import PacmanFiLM, PacmanFiLMConfig


EPOCHS = 200
BATCH_SIZE = 1
NUM_WORKERS = 4

LEARNING_RATE = 1e-3
SCORE_SCALE = 10_000.0

DATASET_PATH = "pacman_dataset"

WANDB_PROJECT = "pacman"
WANDB_RUN_NAME = "pacman-film"

CHECKPOINT_DIR = Path("checkpoints/pacman_film")

HISTOGRAM_EVERY = 100

# The edit region is derived from the structured state transition rather
# than from RGB differences.  A changed state-map cell is an edit cell.
MASK_LOSS_WEIGHT = 1.0
STATE_EDIT_THRESHOLD = 0.0


def build_state_edit_mask(
    batch: dict[str, Any],
) -> torch.Tensor:
    """Project changed state-map cells into image space.

    This is an oracle semantic edit mask: it marks cells whose structured
    state differs between the current and next state.  It is intentionally
    independent of RGB differences so that rendering/anti-aliasing changes
    cannot turn the entire image into an edit region.
    """
    state_map = batch["state_map"]
    state_to_map = batch["state_to_map"]

    state_change = (
        (state_to_map - state_map)
        .abs()
        .amax(dim=1, keepdim=True)
        > STATE_EDIT_THRESHOLD
    ).float()

    image_height = batch["image"].shape[-2]
    image_width = batch["image"].shape[-1]

    return F.interpolate(
        state_change,
        size=(image_height, image_width),
        mode="nearest",
    )


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE normalized by the number of supervised pixels."""
    squared_error = (prediction - target).pow(2)
    mask = mask.expand_as(squared_error)

    return (
        (squared_error * mask).sum()
        / mask.sum().clamp_min(1.0)
    )


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
) -> tuple[torch.Tensor, torch.Tensor]:
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
    proc_batch = data_utils.preprocess_batch(
        batch,
        device=device,
        rows=rows,
        cols=cols,
        score_scale=SCORE_SCALE,
    )

    # Keep every tensor in the processed batch on the selected device.
    # This is especially important for fixed visualization batches,
    # whose DataLoader tensors originate on CPU.
    for key, value in proc_batch.items():
        if torch.is_tensor(value):
            proc_batch[key] = value.to(device)

    return proc_batch


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
        if batch_idx >= 1:
            continue

        proc_batch = prepare_batch(
            batch,
            device=device,
            rows=rows,
            cols=cols,
        )

        optimizer.zero_grad(set_to_none=True)

        pred_image, pred_edit_mask = forward_model(
            model,
            proc_batch,
        )
        raw_pred_image = pred_image

        current_image = proc_batch["image"]
        target_image = proc_batch["target_image"]
        target_delta = target_image - current_image

        # Oracle semantic edit mask from the known state transition.
        target_edit_mask = build_state_edit_mask(
            proc_batch,
        )

        # Hard-gate the predicted residual with the oracle mask.
        # This removes the identity/background shortcut from the image
        # objective while we test whether the image head can learn the
        # actual edit given the correct locality.
        raw_pred_delta = pred_image - current_image
        gated_pred_image = (
            current_image
            + target_edit_mask * raw_pred_delta
        )

        edit_image_loss = masked_mse(
            gated_pred_image,
            target_image,
            target_edit_mask,
        )

        mask_loss = F.binary_cross_entropy(
            pred_edit_mask,
            target_edit_mask,
        )

        loss = (
            edit_image_loss
            + MASK_LOSS_WEIGHT * mask_loss
        )

        loss.backward()

        stats = model_stats(model)

        optimizer.step()

        learning_rate = optimizer.param_groups[0]["lr"]

        # -----------------------------------------------------------
        # Scalar logging.
        # -----------------------------------------------------------

        raw_image_mse = torch.mean(
            (pred_image - target_image) ** 2
        )
        gated_image_mse = torch.mean(
            (gated_pred_image - target_image) ** 2
        )
        edit_delta_mse = masked_mse(
            raw_pred_delta,
            target_delta,
            target_edit_mask,
        )
        keep_raw_mse = masked_mse(
            pred_image,
            current_image,
            1.0 - target_edit_mask,
        )

        wandb_metrics: dict[str, Any] = {
            "train/loss": loss.item(),
            "train/edit_image_loss": edit_image_loss.item(),
            "train/mask_loss": mask_loss.item(),
            "train/raw_image_mse": raw_image_mse.item(),
            "train/gated_image_mse": gated_image_mse.item(),
            "train/edit_delta_mse": edit_delta_mse.item(),
            "train/keep_raw_mse": keep_raw_mse.item(),
            "train/edit_pixel_fraction": target_edit_mask.mean().item(),
            "train/pred_edit_pixel_fraction": (
                (pred_edit_mask > 0.5).float().mean().item()
            ),
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
        # Image / residual statistics.
        # -----------------------------------------------------------

        pred_delta = raw_pred_delta
        gated_pred_delta = gated_pred_image - current_image

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

        for name, value in tensor_stats(
            gated_pred_delta
        ).items():
            wandb_metrics[
                f"train/gated_pred_delta/{name}"
            ] = value

        pred_image = gated_pred_image.clamp(0, 1)

        for name, value in tensor_stats(
            raw_pred_image
        ).items():
            wandb_metrics[
                f"train/raw_pred_image/{name}"
            ] = value

        for name, value in tensor_stats(
            pred_image
        ).items():
            wandb_metrics[
                f"train/gated_pred_image/{name}"
            ] = value

        for name, value in tensor_stats(
            pred_edit_mask
        ).items():
            wandb_metrics[
                f"train/pred_edit_mask/{name}"
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

    pred_image, pred_edit_mask = forward_model(
        model,
        proc_batch,
    )

    current_image_full = proc_batch["image"]
    target_image = proc_batch["target_image"]
    target_delta = target_image - current_image_full

    target_edit_mask = build_state_edit_mask(
        proc_batch,
    )

    raw_pred_image = pred_image
    raw_pred_delta = raw_pred_image - current_image_full
    pred_image = (
        current_image_full
        + target_edit_mask * raw_pred_delta
    ).clamp(0, 1)
    pred_delta = pred_image - current_image_full

    # ---------------------------------------------------------------
    # Only log a few examples.
    # ---------------------------------------------------------------

    # Keep these tensors on the selected device until all metrics have
    # been computed. They are moved to CPU only after metric evaluation.

    # ---------------------------------------------------------------
    # Visualization transforms.
    # ---------------------------------------------------------------

    raw_pred_delta_visual = (
        (raw_pred_delta + 1.0) / 2.0
    ).clamp(0, 1)

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

    raw_delta_mse = torch.mean(
        (raw_pred_delta - target_delta) ** 2
    )

    raw_delta_mae = torch.mean(
        (raw_pred_delta - target_delta).abs()
    )

    delta_mse = torch.mean(
        (pred_delta - target_delta) ** 2
    )

    delta_mae = torch.mean(
        (pred_delta - target_delta).abs()
    )

    raw_image_mse = torch.mean(
        (raw_pred_image - target_image) ** 2
    )

    raw_image_mae = torch.mean(
        (raw_pred_image - target_image).abs()
    )

    image_mse = torch.mean(
        (pred_image - target_image) ** 2
    )

    image_mae = torch.mean(
        (pred_image - target_image).abs()
    )

    edit_image_mse = masked_mse(
        pred_image,
        target_image,
        target_edit_mask,
    )

    keep_image_mse = masked_mse(
        pred_image,
        current_image_full,
        1.0 - target_edit_mask,
    )

    edit_delta_mse = masked_mse(
        pred_delta,
        target_delta,
        target_edit_mask,
    )

    mask_bce = F.binary_cross_entropy(
        pred_edit_mask,
        target_edit_mask,
    )

    pred_mask_binary = pred_edit_mask > 0.5
    target_mask_binary = target_edit_mask > 0.5

    mask_intersection = (
        pred_mask_binary & target_mask_binary
    ).sum().float()
    mask_union = (
        pred_mask_binary | target_mask_binary
    ).sum().float()

    mask_iou = (
        mask_intersection / mask_union.clamp_min(1.0)
    )

    # ---------------------------------------------------------------
    # Move only the tensors used for visualization to CPU.
    #
    # All metrics above intentionally run on the selected training
    # device. Moving tensors to CPU before the metrics causes device
    # mismatch errors when the model is running on XPU/CUDA/MPS.
    # ---------------------------------------------------------------

    current_image = current_image_full[:4].cpu()
    raw_pred_image = raw_pred_image[:4].cpu()
    pred_image = pred_image[:4].cpu()
    target_image = target_image[:4].cpu()

    pred_delta = pred_delta[:4].cpu()
    raw_pred_delta = raw_pred_delta[:4].cpu()
    target_delta = target_delta[:4].cpu()
    pred_edit_mask = pred_edit_mask[:4].cpu()
    target_edit_mask = target_edit_mask[:4].cpu()

    wandb.log(
        {
            "eval/fixed_batch_raw_delta_mse": raw_delta_mse.item(),
            "eval/fixed_batch_raw_delta_mae": raw_delta_mae.item(),
            "eval/fixed_batch_delta_mse": delta_mse.item(),
            "eval/fixed_batch_delta_mae": delta_mae.item(),
            "eval/fixed_batch_raw_image_mse": raw_image_mse.item(),
            "eval/fixed_batch_raw_image_mae": raw_image_mae.item(),
            "eval/fixed_batch_image_mse": image_mse.item(),
            "eval/fixed_batch_image_mae": image_mae.item(),
            "eval/fixed_batch_edit_image_mse": edit_image_mse.item(),
            "eval/fixed_batch_keep_image_mse": keep_image_mse.item(),
            "eval/fixed_batch_edit_delta_mse": edit_delta_mse.item(),
            "eval/fixed_batch_mask_bce": mask_bce.item(),
            "eval/fixed_batch_mask_iou": mask_iou.item(),
            "eval/fixed_batch_edit_pixel_fraction": (
                target_edit_mask.mean().item()
            ),
            "eval/fixed_batch_pred_edit_pixel_fraction": (
                (pred_edit_mask > 0.5).float().mean().item()
            ),

            "images/current": [
                wandb.Image(image)
                for image in current_image
            ],

            "images/raw_prediction": [
                wandb.Image(image)
                for image in raw_pred_image
            ],

            "images/prediction": [
                wandb.Image(image)
                for image in pred_image
            ],

            "images/target": [
                wandb.Image(image)
                for image in target_image
            ],

            "images/raw_predicted_delta": [
                wandb.Image(image)
                for image in raw_pred_delta_visual
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

            "images/predicted_edit_mask": [
                wandb.Image(image)
                for image in pred_edit_mask
            ],

            "images/target_edit_mask": [
                wandb.Image(image)
                for image in target_edit_mask
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

    # Select the best available accelerator automatically.
    # Priority: Intel XPU -> NVIDIA CUDA -> Apple MPS -> CPU.
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

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
        # shuffle=True,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type in {"cuda", "xpu"},
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
        batch_size=1,
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
            "experiment": "overfit test to oracle state-mask gated edit prediction",
            "objective": (
                "test whether the model is expressive to fit to this problem"
            ),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "learning_rate": LEARNING_RATE,
            "score_scale": SCORE_SCALE,
            "state_edit_threshold": STATE_EDIT_THRESHOLD,
            "mask_loss_weight": MASK_LOSS_WEIGHT,
            "mask_source": "state_map_delta",
            "image_gating": "oracle_target_state_mask",
            "image_loss": "masked_edit_mse",
            "dataset_path": DATASET_PATH,
            "rows": rows,
            "cols": cols,
            "device": str(device),
            "model": "PacmanFiLM+EditMask",
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
