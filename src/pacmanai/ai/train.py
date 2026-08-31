from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from pacmanai.dataset import data_utils
from pacmanai.dataset.pacman_dataset import PacmanDataset

from .model import PacmanFiLM, PacmanFiLMConfig

EPOCHS = 20
BATCH_SIZE = 8
NUM_WORKERS = 4

LEARNING_RATE = 2e-3
SCORE_SCALE = 10_000.0

DATASET_PATH = "pacman_dataset_new"

WANDB_PROJECT = "pacman"
WANDB_RUN_NAME = "pacman-film"

CHECKPOINT_DIR = Path("checkpoints/pacman_film")

HISTOGRAM_EVERY = 100

VAL_FRACTION = 0.10
SPLIT_SEED = 42

RECON_LOSS_WEIGHT = 1.0
MASK_LOSS_WEIGHT = 0.1
INCORRECT_EDIT_WEIGHT = 0.1

STATE_EDIT_THRESHOLD = 0.0


def build_state_edit_mask(
    batch: dict[str, Any],
) -> torch.Tensor:
    """Build the target semantic edit mask from the state transition."""

    state_map = batch["state_map"]
    state_to_map = batch["state_to_map"]

    state_change = (
        (state_to_map - state_map).abs().amax(dim=1, keepdim=True)
        > STATE_EDIT_THRESHOLD
    ).float()

    image_height = batch["image"].shape[-2]
    image_width = batch["image"].shape[-1]

    maze_height = round(image_width * state_map.shape[-2] / state_map.shape[-1])

    maze_edit_mask = F.interpolate(
        state_change,
        size=(maze_height, image_width),
        mode="nearest",
    )

    edit_mask = torch.zeros(
        (
            maze_edit_mask.shape[0],
            1,
            image_height,
            image_width,
        ),
        device=maze_edit_mask.device,
        dtype=maze_edit_mask.dtype,
    )

    copy_height = min(
        maze_height,
        image_height,
    )

    edit_mask[..., :copy_height, :] = maze_edit_mask[..., :copy_height, :]

    return edit_mask


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE normalized by the number of selected pixels."""

    squared_error = (prediction - target).pow(2)
    mask = mask.expand_as(squared_error)

    return (squared_error * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MAE normalized by the number of selected pixels."""

    absolute_error = (prediction - target).abs()
    mask = mask.expand_as(absolute_error)

    return (absolute_error * mask).sum() / mask.sum().clamp_min(1.0)


def psnr_from_mse(
    mse: torch.Tensor,
) -> torch.Tensor:
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


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
        "parameter_abs_mean": (parameter_abs_sum / parameter_count),
        "gradient_norm": gradient_sq_sum**0.5,
        "gradient_abs_mean": (
            gradient_abs_sum / gradient_count if gradient_count else 0.0
        ),
    }


def log_model_histograms(
    model: nn.Module,
) -> None:
    for name, parameter in model.named_parameters():
        wandb.log(
            {
                f"parameters/{name}": wandb.Histogram(parameter.detach().cpu().numpy()),
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

    for key, value in proc_batch.items():
        if torch.is_tensor(value):
            proc_batch[key] = value.to(device)

    return proc_batch


def compute_losses(
    *,
    raw_prediction: torch.Tensor,
    predicted_mask: torch.Tensor,
    current_image: torch.Tensor,
    target_image: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute the three-term objective.

    The predicted image is explicitly constructed using the predicted
    mask. The target mask is only used as supervision.

        predicted = current + predicted_mask * (raw_prediction - current)

    Loss:

        recon:
            full-image reconstruction MSE

        mask:
            BCE(predicted_mask, target_mask)

        incorrect_edit:
            reconstruction MSE restricted to pixels where the
            predicted and target masks disagree
    """

    # ---------------------------------------------------------------
    # The model's mask determines where its proposed image edit is
    # actually applied.
    # ---------------------------------------------------------------

    prediction = current_image + predicted_mask * (raw_prediction - current_image)

    # ---------------------------------------------------------------
    # 1. Original reconstruction term.
    #
    # Always evaluates the complete predicted target image.
    # ---------------------------------------------------------------

    reconstruction_loss = torch.mean((prediction - target_image).pow(2))

    # ---------------------------------------------------------------
    # 2. Mask correctness.
    # ---------------------------------------------------------------

    mask_loss = F.binary_cross_entropy(
        predicted_mask,
        target_mask,
    )

    # ---------------------------------------------------------------
    # 3. Incorrect-edit loss.
    #
    # The disagreement mask is high precisely where the predicted
    # edit region does not agree with the target edit region.
    #
    # This prevents the model from making visually incorrect edits
    # outside the actual transition region.
    # ---------------------------------------------------------------

    mask_disagreement = (predicted_mask - target_mask).abs()

    incorrect_edit_loss = masked_mse(
        prediction,
        target_image,
        mask_disagreement,
    )

    total_loss = (
        RECON_LOSS_WEIGHT * reconstruction_loss
        + MASK_LOSS_WEIGHT * mask_loss
        + INCORRECT_EDIT_WEIGHT * incorrect_edit_loss
    )

    return {
        "loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "mask_loss": mask_loss,
        "incorrect_edit_loss": incorrect_edit_loss,
        "prediction": prediction,
        "mask_disagreement": mask_disagreement,
    }


def compute_metrics(
    *,
    raw_prediction: torch.Tensor,
    predicted_mask: torch.Tensor,
    current_image: torch.Tensor,
    target_image: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute reconstruction and mask metrics."""

    prediction = current_image + predicted_mask * (raw_prediction - current_image)

    raw_prediction_mse = torch.mean((raw_prediction - target_image).pow(2))

    raw_prediction_mae = torch.mean((raw_prediction - target_image).abs())

    reconstruction_mse = torch.mean((prediction - target_image).pow(2))

    reconstruction_mae = torch.mean((prediction - target_image).abs())

    identity_mse = torch.mean((current_image - target_image).pow(2))

    # ---------------------------------------------------------------
    # Region metrics.
    # ---------------------------------------------------------------

    edit_mse = masked_mse(
        prediction,
        target_image,
        target_mask,
    )

    edit_mae = masked_mae(
        prediction,
        target_image,
        target_mask,
    )

    keep_mse = masked_mse(
        prediction,
        target_image,
        1.0 - target_mask,
    )

    # How much the model improves over simply copying the input.
    identity_improvement = (identity_mse - reconstruction_mse) / identity_mse.clamp_min(
        1e-12
    )

    # ---------------------------------------------------------------
    # Mask metrics.
    # ---------------------------------------------------------------

    predicted_mask_binary = predicted_mask > 0.5

    target_mask_binary = target_mask > 0.5

    intersection = (predicted_mask_binary & target_mask_binary).sum().float()

    union = (predicted_mask_binary | target_mask_binary).sum().float()

    true_positive = intersection

    false_positive = (predicted_mask_binary & ~target_mask_binary).sum().float()

    false_negative = (~predicted_mask_binary & target_mask_binary).sum().float()

    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)

    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)

    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)

    iou = intersection / union.clamp_min(1.0)

    return {
        "raw_prediction_mse": raw_prediction_mse,
        "raw_prediction_mae": raw_prediction_mae,
        "reconstruction_mse": reconstruction_mse,
        "reconstruction_mae": reconstruction_mae,
        "identity_mse": identity_mse,
        "edit_mse": edit_mse,
        "edit_mae": edit_mae,
        "keep_mse": keep_mse,
        "identity_improvement": identity_improvement,
        "mask_iou": iou,
        "mask_precision": precision,
        "mask_recall": recall,
        "mask_f1": f1,
        "target_edit_pixel_fraction": target_mask.mean(),
        "predicted_edit_pixel_fraction": predicted_mask.mean(),
        "mask_disagreement_fraction": ((predicted_mask - target_mask).abs().mean()),
        "psnr": psnr_from_mse(reconstruction_mse),
        "edit_psnr": psnr_from_mse(edit_mse),
    }


def accumulate_metrics(
    totals: dict[str, float],
    metrics: dict[str, torch.Tensor],
) -> None:
    for name, value in metrics.items():
        totals[name] = totals.get(name, 0.0) + value.detach().item()


def average_metrics(
    totals: dict[str, float],
    count: int,
) -> dict[str, float]:
    if count == 0:
        return {}

    return {name: value / count for name, value in totals.items()}


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
    batch_count = 0

    for batch_idx, batch in enumerate(loader):
        proc_batch = prepare_batch(
            batch,
            device=device,
            rows=rows,
            cols=cols,
        )

        optimizer.zero_grad(set_to_none=True)

        raw_prediction, predicted_mask = forward_model(
            model,
            proc_batch,
        )

        current_image = proc_batch["image"]
        target_image = proc_batch["target_image"]

        target_mask = build_state_edit_mask(
            proc_batch,
        )

        losses = compute_losses(
            raw_prediction=raw_prediction,
            predicted_mask=predicted_mask,
            current_image=current_image,
            target_image=target_image,
            target_mask=target_mask,
        )

        loss = losses["loss"]

        loss.backward()

        stats = model_stats(model)

        optimizer.step()

        metrics = compute_metrics(
            raw_prediction=raw_prediction,
            predicted_mask=predicted_mask,
            current_image=current_image,
            target_image=target_image,
            target_mask=target_mask,
        )

        totals["loss"] = totals.get("loss", 0.0) + loss.detach().item()

        totals["reconstruction_loss"] = (
            totals.get("reconstruction_loss", 0.0)
            + losses["reconstruction_loss"].detach().item()
        )

        totals["mask_loss"] = (
            totals.get("mask_loss", 0.0) + losses["mask_loss"].detach().item()
        )

        totals["incorrect_edit_loss"] = (
            totals.get("incorrect_edit_loss", 0.0)
            + losses["incorrect_edit_loss"].detach().item()
        )

        accumulate_metrics(
            totals,
            metrics,
        )

        batch_count += 1

        learning_rate = optimizer.param_groups[0]["lr"]

        wandb_metrics: dict[str, Any] = {
            "train/loss": loss.item(),
            "train/reconstruction_loss": (losses["reconstruction_loss"].item()),
            "train/mask_loss": (losses["mask_loss"].item()),
            "train/incorrect_edit_loss": (losses["incorrect_edit_loss"].item()),
            "train/learning_rate": learning_rate,
            "train/gradient_norm": stats["gradient_norm"],
            "train/gradient_abs_mean": stats["gradient_abs_mean"],
            "train/parameter_norm": stats["parameter_norm"],
            "train/parameter_abs_mean": stats["parameter_abs_mean"],
        }

        for name, value in metrics.items():
            wandb_metrics[f"train/{name}"] = value.item()

        # Useful specifically for diagnosing raw residual output.
        for name, value in tensor_stats(raw_prediction).items():
            wandb_metrics[f"train/raw_prediction/{name}"] = value

        for name, value in tensor_stats(predicted_mask).items():
            wandb_metrics[f"train/predicted_mask/{name}"] = value

        wandb.log(
            wandb_metrics,
            step=global_step,
        )

        if HISTOGRAM_EVERY > 0 and global_step % HISTOGRAM_EVERY == 0:
            log_model_histograms(model)

        print(
            f"epoch: {epoch:03d} "
            f"batch: {batch_idx + 1:04d}/{len(loader):04d} "
            f"step: {global_step:07d} "
            f"loss: {loss.item():.6f} "
            f"recon: "
            f"{losses['reconstruction_loss'].item():.6f} "
            f"mask: "
            f"{losses['mask_loss'].item():.6f} "
            f"incorrect: "
            f"{losses['incorrect_edit_loss'].item():.6f} "
            f"mask_iou: "
            f"{metrics['mask_iou'].item():.4f}"
        )

        global_step += 1

    return (
        average_metrics(
            totals,
            batch_count,
        ),
        global_step,
    )


@torch.no_grad()
def evaluate(
    model: PacmanFiLM,
    loader: DataLoader,
    *,
    device: torch.device,
    rows: int,
    cols: int,
) -> dict[str, float]:
    model.eval()

    totals: dict[str, float] = {}
    batch_count = 0

    for batch in loader:
        proc_batch = prepare_batch(
            batch,
            device=device,
            rows=rows,
            cols=cols,
        )

        raw_prediction, predicted_mask = forward_model(
            model,
            proc_batch,
        )

        current_image = proc_batch["image"]
        target_image = proc_batch["target_image"]

        target_mask = build_state_edit_mask(
            proc_batch,
        )

        losses = compute_losses(
            raw_prediction=raw_prediction,
            predicted_mask=predicted_mask,
            current_image=current_image,
            target_image=target_image,
            target_mask=target_mask,
        )

        metrics = compute_metrics(
            raw_prediction=raw_prediction,
            predicted_mask=predicted_mask,
            current_image=current_image,
            target_image=target_image,
            target_mask=target_mask,
        )

        totals["loss"] = totals.get("loss", 0.0) + losses["loss"].item()

        totals["reconstruction_loss"] = (
            totals.get("reconstruction_loss", 0.0)
            + losses["reconstruction_loss"].item()
        )

        totals["mask_loss"] = totals.get("mask_loss", 0.0) + losses["mask_loss"].item()

        totals["incorrect_edit_loss"] = (
            totals.get("incorrect_edit_loss", 0.0)
            + losses["incorrect_edit_loss"].item()
        )

        accumulate_metrics(
            totals,
            metrics,
        )

        batch_count += 1

    return average_metrics(
        totals,
        batch_count,
    )


@torch.no_grad()
def log_predictions(
    model: PacmanFiLM,
    batch: dict[str, Any],
    *,
    device: torch.device,
    rows: int,
    cols: int,
    global_step: int,
) -> None:
    model.eval()

    proc_batch = prepare_batch(
        batch,
        device=device,
        rows=rows,
        cols=cols,
    )

    raw_prediction, predicted_mask = forward_model(
        model,
        proc_batch,
    )

    current_image = proc_batch["image"]
    target_image = proc_batch["target_image"]

    target_mask = build_state_edit_mask(
        proc_batch,
    )

    prediction = current_image + predicted_mask * (raw_prediction - current_image)

    raw_delta = raw_prediction - current_image

    predicted_delta = prediction - current_image

    target_delta = target_image - current_image

    delta_error = (predicted_delta - target_delta).abs().clamp(0, 1)

    metrics = compute_metrics(
        raw_prediction=raw_prediction,
        predicted_mask=predicted_mask,
        current_image=current_image,
        target_image=target_image,
        target_mask=target_mask,
    )

    losses = compute_losses(
        raw_prediction=raw_prediction,
        predicted_mask=predicted_mask,
        current_image=current_image,
        target_image=target_image,
        target_mask=target_mask,
    )

    # ---------------------------------------------------------------
    # Visualization.
    #
    # Raw prediction may legitimately be outside [0, 1] because the
    # image head is unconstrained. Keep the raw tensor for metrics,
    # but clamp only the visualization.
    # ---------------------------------------------------------------

    current_visual = current_image.clamp(0, 1)
    raw_prediction_visual = raw_prediction.clamp(0, 1)
    prediction_visual = prediction.clamp(0, 1)
    target_visual = target_image.clamp(0, 1)

    raw_delta_visual = ((raw_delta + 1.0) / 2.0).clamp(0, 1)

    predicted_delta_visual = ((predicted_delta + 1.0) / 2.0).clamp(0, 1)

    target_delta_visual = ((target_delta + 1.0) / 2.0).clamp(0, 1)

    delta_error_visual = (delta_error).clamp(0, 1)

    predicted_delta_abs = (raw_delta.abs()).clamp(0, 1)

    n_images = min(
        4,
        current_image.shape[0],
    )

    current_visual = current_visual[:n_images].detach().cpu()

    raw_prediction_visual = raw_prediction_visual[:n_images].detach().cpu()

    prediction_visual = prediction_visual[:n_images].detach().cpu()

    target_visual = target_visual[:n_images].detach().cpu()

    raw_delta_visual = raw_delta_visual[:n_images].detach().cpu()

    predicted_delta_visual = predicted_delta_visual[:n_images].detach().cpu()

    target_delta_visual = target_delta_visual[:n_images].detach().cpu()

    delta_error_visual = delta_error_visual[:n_images].detach().cpu()

    predicted_delta_abs = predicted_delta_abs[:n_images].detach().cpu()

    predicted_mask_visual = predicted_mask[:n_images].detach().cpu()

    target_mask_visual = target_mask[:n_images].detach().cpu()

    mask_disagreement_visual = (
        (predicted_mask - target_mask).abs()[:n_images].detach().cpu()
    )

    wandb.log(
        {
            # -------------------------------------------------------
            # Fixed validation metrics
            # -------------------------------------------------------
            "eval/fixed_batch_loss": (losses["loss"].item()),
            "eval/fixed_batch_reconstruction_loss": (
                losses["reconstruction_loss"].item()
            ),
            "eval/fixed_batch_mask_loss": (losses["mask_loss"].item()),
            "eval/fixed_batch_incorrect_edit_loss": (
                losses["incorrect_edit_loss"].item()
            ),
            "eval/fixed_batch_reconstruction_mse": (
                metrics["reconstruction_mse"].item()
            ),
            "eval/fixed_batch_edit_mse": (metrics["edit_mse"].item()),
            "eval/fixed_batch_keep_mse": (metrics["keep_mse"].item()),
            "eval/fixed_batch_identity_mse": (metrics["identity_mse"].item()),
            "eval/fixed_batch_identity_improvement": (
                metrics["identity_improvement"].item()
            ),
            "eval/fixed_batch_psnr": (metrics["psnr"].item()),
            "eval/fixed_batch_edit_psnr": (metrics["edit_psnr"].item()),
            # -------------------------------------------------------
            # Validation mask metrics
            # -------------------------------------------------------
            "eval/fixed_batch_mask_iou": (metrics["mask_iou"].item()),
            "eval/fixed_batch_mask_precision": (metrics["mask_precision"].item()),
            "eval/fixed_batch_mask_recall": (metrics["mask_recall"].item()),
            "eval/fixed_batch_mask_f1": (metrics["mask_f1"].item()),
            "eval/fixed_batch_target_edit_pixel_fraction": (
                metrics["target_edit_pixel_fraction"].item()
            ),
            "eval/fixed_batch_predicted_edit_pixel_fraction": (
                metrics["predicted_edit_pixel_fraction"].item()
            ),
            "eval/fixed_batch_mask_disagreement_fraction": (
                metrics["mask_disagreement_fraction"].item()
            ),
            # -------------------------------------------------------
            # Raw prediction diagnostics
            # -------------------------------------------------------
            "eval/fixed_batch_raw_prediction_min": (raw_prediction.min().item()),
            "eval/fixed_batch_raw_prediction_max": (raw_prediction.max().item()),
            # -------------------------------------------------------
            # Images
            # -------------------------------------------------------
            "images/current": [wandb.Image(image) for image in current_visual],
            "images/raw_prediction": [
                wandb.Image(image) for image in raw_prediction_visual
            ],
            "images/prediction": [wandb.Image(image) for image in prediction_visual],
            "images/target": [wandb.Image(image) for image in target_visual],
            "images/raw_predicted_delta": [
                wandb.Image(image) for image in raw_delta_visual
            ],
            "images/predicted_delta": [
                wandb.Image(image) for image in predicted_delta_visual
            ],
            "images/target_delta": [
                wandb.Image(image) for image in target_delta_visual
            ],
            "images/predicted_delta_abs": [
                wandb.Image(image) for image in predicted_delta_abs
            ],
            "images/delta_error": [wandb.Image(image) for image in delta_error_visual],
            # -------------------------------------------------------
            # Mask visualizations — now logged for validation too.
            # -------------------------------------------------------
            "images/predicted_edit_mask": [
                wandb.Image(image) for image in predicted_mask_visual
            ],
            "images/target_edit_mask": [
                wandb.Image(image) for image in target_mask_visual
            ],
            "images/mask_disagreement": [
                wandb.Image(image) for image in mask_disagreement_visual
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

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    metadata = json.loads((Path(DATASET_PATH) / "metadata.json").read_text())

    rows = metadata["rows"]
    cols = metadata["cols"]

    dataset = PacmanDataset(
        DATASET_PATH,
        cache_images=False,
    )

    dataset_size = len(dataset)

    val_size = max(
        1,
        round(dataset_size * VAL_FRACTION),
    )

    train_size = dataset_size - val_size

    if train_size < 1:
        raise RuntimeError("Dataset is too small to create train and validation sets.")

    split_generator = torch.Generator().manual_seed(SPLIT_SEED)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    print(f"Dataset: {dataset_size} samples (train={train_size}, val={val_size})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type in {"cuda", "xpu"},
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type in {"cuda", "xpu"},
        persistent_workers=NUM_WORKERS > 0,
    )

    # Fixed held-out validation examples.
    fixed_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    fixed_batch = next(iter(fixed_loader))

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    config = PacmanFiLMConfig()

    model = PacmanFiLM(config).to(device)

    print(f"Trainable parameters: {model.count_parameters():,}")

    # ---------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
    )

    # ---------------------------------------------------------------
    # W&B
    # ---------------------------------------------------------------

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "experiment": ("PacmanFiLM image reconstruction with learned edit mask"),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "learning_rate": LEARNING_RATE,
            "score_scale": SCORE_SCALE,
            "dataset_path": DATASET_PATH,
            "dataset_size": dataset_size,
            "train_size": train_size,
            "val_size": val_size,
            "val_fraction": VAL_FRACTION,
            "split_seed": SPLIT_SEED,
            "rows": rows,
            "cols": cols,
            "device": str(device),
            "model": "PacmanFiLM",
            "model_parameters": model.count_parameters(),
            "reconstruction_loss_weight": (RECON_LOSS_WEIGHT),
            "mask_loss_weight": (MASK_LOSS_WEIGHT),
            "incorrect_edit_loss_weight": (INCORRECT_EDIT_WEIGHT),
            "loss_formulation": (
                "reconstruction_mse + mask_bce + mask_disagreement_mse"
            ),
            "prediction_formulation": (
                "current + predicted_mask * (raw_prediction - current)"
            ),
        },
    )

    wandb.watch(
        model,
        log="all",
        log_freq=HISTOGRAM_EVERY,
        log_graph=False,
    )

    global_step = 0
    best_val_loss = float("inf")

    try:
        for epoch in range(
            1,
            EPOCHS + 1,
        ):
            # -------------------------------------------------------
            # Train
            # -------------------------------------------------------

            train_metrics, global_step = train_one_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                device=device,
                rows=rows,
                cols=cols,
                epoch=epoch,
                global_step=global_step,
            )

            # -------------------------------------------------------
            # Validation
            # -------------------------------------------------------

            val_metrics = evaluate(
                model,
                val_loader,
                device=device,
                rows=rows,
                cols=cols,
            )

            # -------------------------------------------------------
            # Epoch curves
            # -------------------------------------------------------

            epoch_metrics = {
                "epoch": epoch,
                "epoch/learning_rate": optimizer.param_groups[0]["lr"],
                "epoch/train_loss": train_metrics["loss"],
                "epoch/val_loss": val_metrics["loss"],
                "epoch/train_reconstruction_loss": (
                    train_metrics["reconstruction_loss"]
                ),
                "epoch/val_reconstruction_loss": (val_metrics["reconstruction_loss"]),
                "epoch/train_mask_loss": (train_metrics["mask_loss"]),
                "epoch/val_mask_loss": (val_metrics["mask_loss"]),
                "epoch/train_incorrect_edit_loss": (
                    train_metrics["incorrect_edit_loss"]
                ),
                "epoch/val_incorrect_edit_loss": (val_metrics["incorrect_edit_loss"]),
                "epoch/train_mask_iou": (train_metrics["mask_iou"]),
                "epoch/val_mask_iou": (val_metrics["mask_iou"]),
                "epoch/train_mask_f1": (train_metrics["mask_f1"]),
                "epoch/val_mask_f1": (val_metrics["mask_f1"]),
                "epoch/train_edit_mse": (train_metrics["edit_mse"]),
                "epoch/val_edit_mse": (val_metrics["edit_mse"]),
                "epoch/train_psnr": (train_metrics["psnr"]),
                "epoch/val_psnr": (val_metrics["psnr"]),
                "epoch/train_identity_improvement": (
                    train_metrics["identity_improvement"]
                ),
                "epoch/val_identity_improvement": (val_metrics["identity_improvement"]),
            }

            wandb.log(
                epoch_metrics,
                step=global_step,
            )

            # Full metric groups.
            wandb.log(
                {f"train_epoch/{name}": value for name, value in train_metrics.items()},
                step=global_step,
            )

            wandb.log(
                {f"val/{name}": value for name, value in val_metrics.items()},
                step=global_step,
            )

            # -------------------------------------------------------
            # Held-out validation visualizations.
            # -------------------------------------------------------

            log_predictions(
                model,
                fixed_batch,
                device=device,
                rows=rows,
                cols=cols,
                global_step=global_step,
            )

            # -------------------------------------------------------
            # Checkpoints.
            # -------------------------------------------------------

            save_checkpoint(
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                loss=val_metrics["loss"],
                path=(CHECKPOINT_DIR / f"pacman-film-{epoch:03d}.pt"),
            )

            save_checkpoint(
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                loss=val_metrics["loss"],
                path=(CHECKPOINT_DIR / "last.pt"),
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]

                save_checkpoint(
                    model,
                    optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    loss=val_metrics["loss"],
                    path=(CHECKPOINT_DIR / "best.pt"),
                )

            print(
                f"\n"
                f"epoch {epoch:03d}/{EPOCHS} | "
                f"train={train_metrics['loss']:.6f} | "
                f"val={val_metrics['loss']:.6f} | "
                f"val recon="
                f"{val_metrics['reconstruction_loss']:.6f} | "
                f"val mask="
                f"{val_metrics['mask_loss']:.6f} | "
                f"val incorrect="
                f"{val_metrics['incorrect_edit_loss']:.6f} | "
                f"val IoU="
                f"{val_metrics['mask_iou']:.4f} | "
                f"val edit MSE="
                f"{val_metrics['edit_mse']:.6f} | "
                f"val PSNR="
                f"{val_metrics['psnr']:.3f}"
            )

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
