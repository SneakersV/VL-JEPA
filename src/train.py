import math
import os

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from losses import VLJepaLoss


def build_pretrain_target(sample):
    """
    Stage 1: query-free pretraining target.

    Since this dataset is VQA, not pure caption data, we create a caption-like
    target from the answer + explanation + cultural information.
    """
    parts = []

    answer = sample.get("answer", "")
    explanation = sample.get("detailed_explanation", "")
    cultural = sample.get("cultural_significance", "")

    if answer:
        parts.append(str(answer))

    if explanation:
        parts.append(str(explanation))

    if cultural:
        parts.append(str(cultural))

    if len(parts) == 0:
        parts.append("No description available.")

    return " ".join(parts)


def build_sft_target(sample):
    """
    Stage 2: supervised VQA target.
    """
    answer = sample.get("answer", "")
    explanation = sample.get("detailed_explanation", "")

    if explanation:
        return f"{answer}. {explanation}"

    return str(answer)


def collate_pretrain(batch):
    """
    Stage 1: query-free / caption-style pretraining.

    Query is fixed and neutral. Target is caption-like text.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    queries_text = [""] * len(batch)
    target_text = [build_pretrain_target(item) for item in batch]

    return {
        "pixel_values": images,
        "queries_text": queries_text,
        "target_text": target_text,
    }


def collate_sft(batch):
    """
    Stage 2: query-conditioned VQA supervised finetuning.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)

    queries_text = [
        item["question"] if item["question"] else "Answer the question about this image."
        for item in batch
    ]

    target_text = [build_sft_target(item) for item in batch]

    return {
        "pixel_values": images,
        "queries_text": queries_text,
        "target_text": target_text,
    }


def prepare_pixel_values(pixel_values, model, device):
    """
    Convert image tensors to the expected V-JEPA input style.

    VietCulturalDataset returns [B, C, H, W]. If values are 0-255, convert to
    0-1, then apply processor mean/std normalization.
    """
    pixel_values = pixel_values.to(device, non_blocking=True).float()

    if pixel_values.max() > 2.0:
        pixel_values = pixel_values / 255.0

    image_mean = getattr(model.vision_processor, "image_mean", [0.485, 0.456, 0.406])
    image_std = getattr(model.vision_processor, "image_std", [0.229, 0.224, 0.225])

    mean = torch.tensor(image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(image_std, device=device).view(1, 3, 1, 1)

    return (pixel_values - mean) / std


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_amp_dtype(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_optimizer(
    model,
    criterion,
    base_lr=5e-5,
    weight_decay=0.01,
    y_encoder_lr_mult=0.05,
):
    y_encoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "y_encoder" in name:
            y_encoder_params.append(param)
        else:
            other_params.append(param)

    loss_params = [param for param in criterion.parameters() if param.requires_grad]
    param_groups = []

    if len(other_params) > 0:
        param_groups.append({
            "params": other_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
        })

    if len(y_encoder_params) > 0:
        param_groups.append({
            "params": y_encoder_params,
            "lr": base_lr * y_encoder_lr_mult,
            "weight_decay": weight_decay,
        })

    if len(loss_params) > 0:
        param_groups.append({
            "params": loss_params,
            "lr": base_lr,
            "weight_decay": 0.0,
        })

    if len(param_groups) == 0:
        raise ValueError("No trainable parameters found for optimizer.")

    return torch.optim.AdamW(param_groups)


def build_scheduler(
    optimizer,
    schedule_type,
    num_training_steps,
    warmup_steps=0,
):
    if schedule_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )

    if schedule_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(num_training_steps, 1),
        )

    if schedule_type is None:
        return None

    raise ValueError(f"Unknown schedule_type '{schedule_type}'.")


def _mean_logs(total_logs, total_steps):
    return {
        key: value / max(total_steps, 1)
        for key, value in total_logs.items()
    }


def _train_batches(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
    epoch,
    stage_name,
    grad_accum_steps=1,
    max_grad_norm=1.0,
    max_batches=None,
):
    model.train()
    criterion.train()

    total_logs = {
        "loss": 0.0,
        "info_nce": 0.0,
        "regularization": 0.0,
    }
    total_steps = 0
    optimizer_updates = 0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc=f"{stage_name} | epoch {epoch}")
    amp_dtype = get_amp_dtype(device)
    use_amp = device.type == "cuda"

    for step, batch in enumerate(pbar):
        if max_batches is not None and step >= max_batches:
            break

        pixel_values = prepare_pixel_values(batch["pixel_values"], model, device)
        queries_text = batch["queries_text"]
        target_text = batch["target_text"]

        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            predicted_embeds, target_embeds = model(
                pixel_values=pixel_values,
                queries_text=queries_text,
                target_text=target_text,
            )

            loss, logs = criterion(predicted_embeds, target_embeds)
            loss_to_backward = loss / grad_accum_steps

        loss_to_backward.backward()

        should_step = (step + 1) % grad_accum_steps == 0
        is_last_batch = step + 1 == len(dataloader)
        hit_batch_limit = max_batches is not None and step + 1 >= max_batches

        if should_step or is_last_batch or hit_batch_limit:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)
            optimizer_updates += 1

        for key in total_logs:
            total_logs[key] += logs[key].detach().float().item()

        total_steps += 1
        pbar.set_postfix(_mean_logs(total_logs, total_steps))

    metrics = _mean_logs(total_logs, total_steps)
    metrics["optimizer_updates"] = optimizer_updates
    metrics["batches"] = total_steps

    return metrics


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device=None,
    stage_name="val",
    max_batches=None,
):
    model.eval()
    criterion.eval()

    if device is None:
        device = get_default_device()

    total_logs = {
        "loss": 0.0,
        "info_nce": 0.0,
        "regularization": 0.0,
    }
    total_steps = 0

    pbar = tqdm(dataloader, desc=stage_name)
    amp_dtype = get_amp_dtype(device)
    use_amp = device.type == "cuda"

    for step, batch in enumerate(pbar):
        if max_batches is not None and step >= max_batches:
            break

        pixel_values = prepare_pixel_values(batch["pixel_values"], model, device)
        queries_text = batch["queries_text"]
        target_text = batch["target_text"]

        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            predicted_embeds, target_embeds = model(
                pixel_values=pixel_values,
                queries_text=queries_text,
                target_text=target_text,
            )

            _, logs = criterion(predicted_embeds, target_embeds)

        for key in total_logs:
            total_logs[key] += logs[key].detach().float().item()

        total_steps += 1
        pbar.set_postfix(_mean_logs(total_logs, total_steps))

    metrics = _mean_logs(total_logs, total_steps)
    metrics["batches"] = total_steps

    return metrics


def _save_checkpoint(
    model,
    criterion,
    optimizer,
    scheduler,
    output_dir,
    tag,
    epoch,
    global_step,
    metrics,
):
    checkpoint_dir = os.path.join(output_dir, tag)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(checkpoint_dir)
    else:
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_state.pt"))

    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
        "criterion_state_dict": criterion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))

    return checkpoint_dir


def _num_update_steps(total_batches, grad_accum_steps):
    return max(math.ceil(total_batches / max(grad_accum_steps, 1)), 1)


def _run_stage(
    model,
    train_dataset,
    val_dataset=None,
    output_dir="checkpoints/stage",
    stage_name="stage",
    collate_fn=collate_sft,
    epochs=1,
    batch_size=2,
    eval_batch_size=None,
    grad_accum_steps=1,
    base_lr=5e-5,
    weight_decay=0.01,
    y_encoder_lr_mult=0.05,
    warmup_steps=0,
    schedule_type="cosine",
    init_temperature=0.07,
    l2_reg_weight=1e-4,
    max_grad_norm=1.0,
    num_workers=0,
    pin_memory=True,
    device=None,
    max_steps=None,
    eval_max_steps=None,
    save_every_epoch=True,
):
    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    model.to(device)

    criterion = VLJepaLoss(
        init_temperature=init_temperature,
        l2_reg_weight=l2_reg_weight,
    ).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == "cuda",
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size or batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and device.type == "cuda",
            collate_fn=collate_fn,
        )

    optimizer = build_optimizer(
        model=model,
        criterion=criterion,
        base_lr=base_lr,
        weight_decay=weight_decay,
        y_encoder_lr_mult=y_encoder_lr_mult,
    )

    total_batches = len(train_loader) * epochs
    if max_steps is not None:
        total_batches = min(total_batches, max_steps)

    scheduler = build_scheduler(
        optimizer=optimizer,
        schedule_type=schedule_type,
        num_training_steps=_num_update_steps(total_batches, grad_accum_steps),
        warmup_steps=warmup_steps,
    )

    os.makedirs(output_dir, exist_ok=True)

    history = []
    global_step = 0
    best_val_loss = None

    for epoch in range(1, epochs + 1):
        if max_steps is None:
            epoch_max_batches = None
        else:
            remaining_steps = max_steps - global_step
            if remaining_steps <= 0:
                break
            epoch_max_batches = min(len(train_loader), remaining_steps)

        train_metrics = _train_batches(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            stage_name=stage_name,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            max_batches=epoch_max_batches,
        )

        global_step += train_metrics["batches"]

        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
        }

        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                stage_name=f"{stage_name} | val",
                max_batches=eval_max_steps,
            )
            epoch_record["val"] = val_metrics

            val_loss = val_metrics["loss"]
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=output_dir,
                    tag="best",
                    epoch=epoch,
                    global_step=global_step,
                    metrics=epoch_record,
                )

        if save_every_epoch:
            _save_checkpoint(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                output_dir=output_dir,
                tag=f"epoch_{epoch}",
                epoch=epoch,
                global_step=global_step,
                metrics=epoch_record,
            )

        _save_checkpoint(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            tag="latest",
            epoch=epoch,
            global_step=global_step,
            metrics=epoch_record,
        )

        history.append(epoch_record)

    return history


def train_pretrain_stage(
    model,
    train_dataset,
    val_dataset=None,
    output_dir="checkpoints/pretrain",
    **kwargs,
):
    return _run_stage(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        stage_name="pretrain",
        collate_fn=collate_pretrain,
        schedule_type=kwargs.pop("schedule_type", "constant"),
        **kwargs,
    )


def train_sft_stage(
    model,
    train_dataset,
    val_dataset=None,
    output_dir="checkpoints/sft",
    **kwargs,
):
    return _run_stage(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        stage_name="sft",
        collate_fn=collate_sft,
        schedule_type=kwargs.pop("schedule_type", "cosine"),
        **kwargs,
    )


def train_one_step_mixed(
    model,
    train_dataset,
    val_dataset=None,
    output_dir="checkpoints/one_step",
    **kwargs,
):
    return _run_stage(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        stage_name="one_step",
        collate_fn=collate_sft,
        schedule_type=kwargs.pop("schedule_type", "cosine"),
        **kwargs,
    )


def run_training_pipeline(
    model,
    train_dataset,
    val_dataset=None,
    run_pretrain=True,
    run_sft=True,
    run_one_step=False,
    pretrain_kwargs=None,
    sft_kwargs=None,
    one_step_kwargs=None,
):
    histories = {}

    if run_pretrain:
        histories["pretrain"] = train_pretrain_stage(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            **(pretrain_kwargs or {}),
        )

    if run_sft:
        histories["sft"] = train_sft_stage(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            **(sft_kwargs or {}),
        )

    if run_one_step:
        histories["one_step"] = train_one_step_mixed(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            **(one_step_kwargs or {}),
        )

    return histories
