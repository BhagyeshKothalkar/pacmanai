from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from pacmanai.dataset.pacman_dataset import PacmanDataset
from pacmanai.dataset import data_utils
from .model import PacmanFiLM, PacmanFiLMConfig


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

BATCH_SIZE = 8
NUM_WORKERS = 4

SCORE_SCALE = 10_000.0

# This must be a completely separate test dataset.
TEST_DATASET_PATH = "pacman_dataset_test"

# Change this to the checkpoint you want to evaluate.
CHECKPOINT_PATH = (
    "checkpoints/pacman_film/best.pt"
)

WANDB_PROJECT = "pacman"
WANDB_RUN_NAME = "pacman-film-test"

STATE_EDIT_THRESHOLD = 0.0


# -------------------------------------------------------------------
# Edit mask
# -------------------------------------------------------------------

def build_state_edit_mask(
    batch: dict[str, Any],
) -> torch.Tensor:
    """Build the target semantic edit mask from the state transition."""

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

    maze_height = round(
        image_width * state_map.shape[-2] / state_map.shape[-1]
    )

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

    edit_mask[..., :copy_height, :] = (
        maze_edit_mask[..., :copy_height, :]
    )

    return edit_mask


# -------------------------------------------------------------------
# Metrics / losses
# -------------------------------------------------------------------

def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE normalized by the number of selected pixels."""

    squared_error = (prediction - target).pow(2)
    mask = mask.expand_as(squared_error)

    return (
        (squared_error * mask).sum()
        / mask.sum().clamp_min(1.0)
    )


def masked_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MAE normalized by the number of selected pixels."""

    absolute_error = (prediction - target).abs()
    mask = mask.expand_as(absolute_error)

    return (
        (absolute_error * mask).sum()
        / mask.sum().clamp_min(1.0)
    )


def psnr_from_mse(
    mse: torch.Tensor,
) -> torch.Tensor:
    return 10.0 * torch.log10(
        1.0 / mse.clamp_min(1e-12)
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
    """Compute the same objective used during training/validation."""

    prediction = (
        current_image
        + predicted_mask
        * (raw_prediction - current_image)
    )

    reconstruction_loss = torch.mean(
        (prediction - target_image).pow(2)
    )

    mask_loss = F.binary_cross_entropy(
        predicted_mask,
        target_mask,
    )

    mask_disagreement = (
        predicted_mask - target_mask
    ).abs()

    incorrect_edit_loss = masked_mse(
        prediction,
        target_image,
        mask_disagreement,
    )

    total_loss = (
        1.0 * reconstruction_loss
        + 0.1 * mask_loss
        + 0.1 * incorrect_edit_loss
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
    """Compute the exact same metrics used by validation."""

    prediction = (
        current_image
        + predicted_mask
        * (raw_prediction - current_image)
    )

    raw_prediction_mse = torch.mean(
        (raw_prediction - target_image).pow(2)
    )

    raw_prediction_mae = torch.mean(
        (raw_prediction - target_image).abs()
    )

    reconstruction_mse = torch.mean(
        (prediction - target_image).pow(2)
    )

    reconstruction_mae = torch.mean(
        (prediction - target_image).abs()
    )

    identity_mse = torch.mean(
        (current_image - target_image).pow(2)
    )

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

    identity_improvement = (
        identity_mse - reconstruction_mse
    ) / identity_mse.clamp_min(1e-12)

    predicted_mask_binary = (
        predicted_mask > 0.5
    )

    target_mask_binary = (
        target_mask > 0.5
    )

    intersection = (
        predicted_mask_binary
        & target_mask_binary
    ).sum().float()

    union = (
        predicted_mask_binary
        | target_mask_binary
    ).sum().float()

    true_positive = intersection

    false_positive = (
        predicted_mask_binary
        & ~target_mask_binary
    ).sum().float()

    false_negative = (
        ~predicted_mask_binary
        & target_mask_binary
    ).sum().float()

    precision = (
        true_positive
        / (true_positive + false_positive).clamp_min(1.0)
    )

    recall = (
        true_positive
        / (true_positive + false_negative).clamp_min(1.0)
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall).clamp_min(1e-12)
    )

    iou = (
        intersection
        / union.clamp_min(1.0)
    )

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
        "mask_disagreement_fraction": (
            (predicted_mask - target_mask)
            .abs()
            .mean()
        ),
        "psnr": psnr_from_mse(
            reconstruction_mse
        ),
        "edit_psnr": psnr_from_mse(
            edit_mse
        ),
    }


def accumulate_metrics(
    totals: dict[str, float],
    metrics: dict[str, torch.Tensor],
) -> None:
    for name, value in metrics.items():
        totals[name] = (
            totals.get(name, 0.0)
            + value.detach().item()
        )


def average_metrics(
    totals: dict[str, float],
    count: int,
) -> dict[str, float]:
    if count == 0:
        return {}

    return {
        name: value / count
        for name, value in totals.items()
    }


# -------------------------------------------------------------------
# Test evaluation
# -------------------------------------------------------------------

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

    for batch_idx, batch in enumerate(loader):
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

        totals["loss"] = (
            totals.get("loss", 0.0)
            + losses["loss"].item()
        )

        totals["reconstruction_loss"] = (
            totals.get("reconstruction_loss", 0.0)
            + losses["reconstruction_loss"].item()
        )

        totals["mask_loss"] = (
            totals.get("mask_loss", 0.0)
            + losses["mask_loss"].item()
        )

        totals["incorrect_edit_loss"] = (
            totals.get("incorrect_edit_loss", 0.0)
            + losses["incorrect_edit_loss"].item()
        )

        accumulate_metrics(
            totals,
            metrics,
        )

        batch_count += 1

        print(
            f"test batch: "
            f"{batch_idx + 1:04d}/{len(loader):04d} "
            f"loss: {losses['loss'].item():.6f} "
            f"recon: "
            f"{losses['reconstruction_loss'].item():.6f} "
            f"mask: "
            f"{losses['mask_loss'].item():.6f} "
            f"incorrect: "
            f"{losses['incorrect_edit_loss'].item():.6f} "
            f"mask_iou: "
            f"{metrics['mask_iou'].item():.4f}"
        )

    return average_metrics(
        totals,
        batch_count,
    )


# -------------------------------------------------------------------
# Fixed test examples
# -------------------------------------------------------------------

@torch.no_grad()
def log_test_predictions(
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

    prediction = (
        current_image
        + predicted_mask
        * (raw_prediction - current_image)
    )

    raw_delta = (
        raw_prediction - current_image
    )

    predicted_delta = (
        prediction - current_image
    )

    target_delta = (
        target_image - current_image
    )

    delta_error = (
        predicted_delta - target_delta
    ).abs().clamp(0, 1)

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
    # Visualization
    # ---------------------------------------------------------------

    current_visual = current_image.clamp(0, 1)
    raw_prediction_visual = raw_prediction.clamp(0, 1)
    prediction_visual = prediction.clamp(0, 1)
    target_visual = target_image.clamp(0, 1)

    raw_delta_visual = (
        (raw_delta + 1.0) / 2.0
    ).clamp(0, 1)

    predicted_delta_visual = (
        (predicted_delta + 1.0) / 2.0
    ).clamp(0, 1)

    target_delta_visual = (
        (target_delta + 1.0) / 2.0
    ).clamp(0, 1)

    delta_error_visual = (
        delta_error
    ).clamp(0, 1)

    predicted_delta_abs = (
        raw_delta.abs()
    ).clamp(0, 1)

    n_images = min(
        4,
        current_image.shape[0],
    )

    current_visual = (
        current_visual[:n_images]
        .detach()
        .cpu()
    )

    raw_prediction_visual = (
        raw_prediction_visual[:n_images]
        .detach()
        .cpu()
    )

    prediction_visual = (
        prediction_visual[:n_images]
        .detach()
        .cpu()
    )

    target_visual = (
        target_visual[:n_images]
        .detach()
        .cpu()
    )

    raw_delta_visual = (
        raw_delta_visual[:n_images]
        .detach()
        .cpu()
    )

    predicted_delta_visual = (
        predicted_delta_visual[:n_images]
        .detach()
        .cpu()
    )

    target_delta_visual = (
        target_delta_visual[:n_images]
        .detach()
        .cpu()
    )

    delta_error_visual = (
        delta_error_visual[:n_images]
        .detach()
        .cpu()
    )

    predicted_delta_abs = (
        predicted_delta_abs[:n_images]
        .detach()
        .cpu()
    )

    predicted_mask_visual = (
        predicted_mask[:n_images]
        .detach()
        .cpu()
    )

    target_mask_visual = (
        target_mask[:n_images]
        .detach()
        .cpu()
    )

    mask_disagreement_visual = (
        (predicted_mask - target_mask)
        .abs()
        [:n_images]
        .detach()
        .cpu()
    )

    wandb.log(
        {
            # -------------------------------------------------------
            # Fixed test metrics
            # -------------------------------------------------------

            "test/fixed_batch_loss": (
                losses["loss"].item()
            ),
            "test/fixed_batch_reconstruction_loss": (
                losses["reconstruction_loss"].item()
            ),
            "test/fixed_batch_mask_loss": (
                losses["mask_loss"].item()
            ),
            "test/fixed_batch_incorrect_edit_loss": (
                losses["incorrect_edit_loss"].item()
            ),
            "test/fixed_batch_reconstruction_mse": (
                metrics["reconstruction_mse"].item()
            ),
            "test/fixed_batch_edit_mse": (
                metrics["edit_mse"].item()
            ),
            "test/fixed_batch_keep_mse": (
                metrics["keep_mse"].item()
            ),
            "test/fixed_batch_identity_mse": (
                metrics["identity_mse"].item()
            ),
            "test/fixed_batch_identity_improvement": (
                metrics["identity_improvement"].item()
            ),
            "test/fixed_batch_psnr": (
                metrics["psnr"].item()
            ),
            "test/fixed_batch_edit_psnr": (
                metrics["edit_psnr"].item()
            ),

            # -------------------------------------------------------
            # Test mask metrics
            # -------------------------------------------------------

            "test/fixed_batch_mask_iou": (
                metrics["mask_iou"].item()
            ),
            "test/fixed_batch_mask_precision": (
                metrics["mask_precision"].item()
            ),
            "test/fixed_batch_mask_recall": (
                metrics["mask_recall"].item()
            ),
            "test/fixed_batch_mask_f1": (
                metrics["mask_f1"].item()
            ),
            "test/fixed_batch_target_edit_pixel_fraction": (
                metrics[
                    "target_edit_pixel_fraction"
                ].item()
            ),
            "test/fixed_batch_predicted_edit_pixel_fraction": (
                metrics[
                    "predicted_edit_pixel_fraction"
                ].item()
            ),
            "test/fixed_batch_mask_disagreement_fraction": (
                metrics[
                    "mask_disagreement_fraction"
                ].item()
            ),

            # -------------------------------------------------------
            # Raw prediction diagnostics
            # -------------------------------------------------------

            "test/fixed_batch_raw_prediction_min": (
                raw_prediction.min().item()
            ),
            "test/fixed_batch_raw_prediction_max": (
                raw_prediction.max().item()
            ),

            # -------------------------------------------------------
            # Images
            # -------------------------------------------------------

            "test_images/current": [
                wandb.Image(image)
                for image in current_visual
            ],

            "test_images/raw_prediction": [
                wandb.Image(image)
                for image in raw_prediction_visual
            ],

            "test_images/prediction": [
                wandb.Image(image)
                for image in prediction_visual
            ],

            "test_images/target": [
                wandb.Image(image)
                for image in target_visual
            ],

            "test_images/raw_predicted_delta": [
                wandb.Image(image)
                for image in raw_delta_visual
            ],

            "test_images/predicted_delta": [
                wandb.Image(image)
                for image in predicted_delta_visual
            ],

            "test_images/target_delta": [
                wandb.Image(image)
                for image in target_delta_visual
            ],

            "test_images/predicted_delta_abs": [
                wandb.Image(image)
                for image in predicted_delta_abs
            ],

            "test_images/delta_error": [
                wandb.Image(image)
                for image in delta_error_visual
            ],

            # -------------------------------------------------------
            # Mask visualizations
            # -------------------------------------------------------

            "test_images/predicted_edit_mask": [
                wandb.Image(image)
                for image in predicted_mask_visual
            ],

            "test_images/target_edit_mask": [
                wandb.Image(image)
                for image in target_mask_visual
            ],

            "test_images/mask_disagreement": [
                wandb.Image(image)
                for image in mask_disagreement_visual
            ],
        },
        step=global_step,
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    # ---------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------

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
            Path(TEST_DATASET_PATH)
            / "metadata.json"
        ).read_text()
    )

    rows = metadata["rows"]
    cols = metadata["cols"]

    dataset = PacmanDataset(
        TEST_DATASET_PATH,
        cache_images=False,
    )

    dataset_size = len(dataset)

    if dataset_size < 1:
        raise RuntimeError(
            "Test dataset is empty."
        )

    print(
        f"Test dataset: "
        f"{dataset_size} samples"
    )

    test_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type in {"cuda", "xpu"},
        persistent_workers=NUM_WORKERS > 0,
    )

    # Fixed held-out test examples.
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

    print(
        f"Trainable parameters: "
        f"{model.count_parameters():,}"
    )

    # ---------------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------------

    checkpoint_path = Path(
        CHECKPOINT_PATH
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    print(
        f"Loading checkpoint: "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    checkpoint_epoch = checkpoint.get(
        "epoch",
        None,
    )

    checkpoint_step = checkpoint.get(
        "global_step",
        None,
    )

    checkpoint_loss = checkpoint.get(
        "loss",
        None,
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint_epoch}"
    )

    print(
        f"Checkpoint global step: "
        f"{checkpoint_step}"
    )

    print(
        f"Checkpoint loss: "
        f"{checkpoint_loss}"
    )

    # ---------------------------------------------------------------
    # W&B
    # ---------------------------------------------------------------

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "experiment": (
                "PacmanFiLM independent test evaluation"
            ),
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "score_scale": SCORE_SCALE,
            "dataset_path": TEST_DATASET_PATH,
            "dataset_size": dataset_size,
            "checkpoint_path": str(
                checkpoint_path
            ),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_global_step": checkpoint_step,
            "checkpoint_loss": checkpoint_loss,
            "rows": rows,
            "cols": cols,
            "device": str(device),
            "model": "PacmanFiLM",
            "model_parameters": (
                model.count_parameters()
            ),
            "reconstruction_loss_weight": 1.0,
            "mask_loss_weight": 0.1,
            "incorrect_edit_loss_weight": 0.1,
            "loss_formulation": (
                "reconstruction_mse + "
                "mask_bce + "
                "mask_disagreement_mse"
            ),
            "prediction_formulation": (
                "current + "
                "predicted_mask * "
                "(raw_prediction - current)"
            ),
        },
    )

    # ---------------------------------------------------------------
    # Test
    # ---------------------------------------------------------------

    try:
        test_metrics = evaluate(
            model,
            test_loader,
            device=device,
            rows=rows,
            cols=cols,
        )

        # -----------------------------------------------------------
        # Full metric group.
        #
        # This mirrors:
        #
        #     wandb.log(
        #         {
        #             f"val/{name}": value
        #             ...
        #         }
        #     )
        #
        # except that this is an independent test set.
        # -----------------------------------------------------------

        wandb.log(
            {
                f"test/{name}": value
                for name, value in test_metrics.items()
            },
            step=0,
        )

        # -----------------------------------------------------------
        # Fixed test examples.
        # -----------------------------------------------------------

        log_test_predictions(
            model,
            fixed_batch,
            device=device,
            rows=rows,
            cols=cols,
            global_step=0,
        )

        # -----------------------------------------------------------
        # Console summary.
        # -----------------------------------------------------------

        print(
            f"\n"
            f"TEST RESULTS\n"
            f"------------\n"
            f"samples="
            f"{dataset_size}\n"
            f"loss="
            f"{test_metrics['loss']:.6f}\n"
            f"recon="
            f"{test_metrics['reconstruction_loss']:.6f}\n"
            f"mask="
            f"{test_metrics['mask_loss']:.6f}\n"
            f"incorrect="
            f"{test_metrics['incorrect_edit_loss']:.6f}\n"
            f"IoU="
            f"{test_metrics['mask_iou']:.4f}\n"
            f"F1="
            f"{test_metrics['mask_f1']:.4f}\n"
            f"edit MSE="
            f"{test_metrics['edit_mse']:.6f}\n"
            f"keep MSE="
            f"{test_metrics['keep_mse']:.6f}\n"
            f"identity MSE="
            f"{test_metrics['identity_mse']:.6f}\n"
            f"identity improvement="
            f"{test_metrics['identity_improvement']:.4f}\n"
            f"PSNR="
            f"{test_metrics['psnr']:.3f}\n"
            f"edit PSNR="
            f"{test_metrics['edit_psnr']:.3f}"
        )

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
