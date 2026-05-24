# VL-JEPA Vietnamese Cultural VQA

This repository trains a VL-JEPA-style vision-language embedding model on the Vietnamese Cultural VQA dataset. The project is notebook-friendly: there is no CLI entrypoint yet, so users import the dataset, model, training, evaluation, and inference helpers directly from Python.

The current model learns an embedding space for image/query inputs and target text. It does not generate natural-language answers directly. For inference, use the predicted embedding for retrieval, ranking, or similarity against candidate answer embeddings.

## Files

- `model.py`: VL-JEPA model definition with frozen V-JEPA vision encoder, Qwen predictor layers, and EmbeddingGemma target encoder.
- `QLoRA_setup.py`: QLoRA model builder, BitsAndBytes config, LoRA config, and trainable-parameter reporting.
- `dataset.py`: Hugging Face dataset loading helpers and `VietCulturalDataset`.
- `train.py`: collators, pixel preprocessing, training loops, evaluation, checkpointing, and all public training functions.
- `losses.py`: InfoNCE and VL-JEPA loss.
- `TRAINING_STRUCTURE.md`: concise training-stage structure.

## Environment Setup

Install PyTorch for your CUDA version first. For example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the rest:

```bash
pip install transformers peft bitsandbytes accelerate huggingface_hub pillow requests tqdm numpy
```

You need network access for the first run because model weights and the dataset are downloaded from Hugging Face.

## Load Data

The default dataset repo is `Dangindev/viet-cultural-vqa`. The loader supports:

- `train` -> `splits/train_data.json`
- `validation` or `val` -> `splits/val_data.json`
- `test` -> `splits/test_data.json`

Images are downloaded manually from the Hugging Face dataset repo. Samples with failed image downloads are skipped.

```python
from dataset import build_viet_cultural_dataset, build_viet_cultural_datasets

train_dataset = build_viet_cultural_dataset(
    split="train",
    n_samples=20,
    image_size=(256, 256),
    normalize=False,
)

val_dataset = build_viet_cultural_dataset(
    split="validation",
    n_samples=8,
    image_size=(256, 256),
    normalize=False,
)

test_dataset = build_viet_cultural_dataset(
    split="test",
    n_samples=8,
    image_size=(256, 256),
    normalize=False,
)
```

You can also load all splits at once:

```python
datasets = build_viet_cultural_datasets(
    train_samples=20,
    val_samples=8,
    test_samples=8,
    image_size=(256, 256),
    normalize=False,
)

train_dataset = datasets["train"]
val_dataset = datasets["validation"]
test_dataset = datasets["test"]
```

## Build the Model

```python
from QLoRA_setup import build_qlora_model

model = build_qlora_model()
```

By default:

- V-JEPA vision encoder is frozen.
- Qwen base weights are loaded in 4-bit.
- LoRA adapters train the Qwen-style attention and MLP modules.
- `vision_proj`, `predictor_head`, and `y_encoder_head` are saved with the adapter.
- The Y encoder body is frozen by default; only `y_encoder_head` remains trainable.

## Quick Smoke Test

Use tiny data and `max_steps=1` to verify the pipeline before a real run:

```python
from dataset import build_viet_cultural_dataset
from QLoRA_setup import build_qlora_model
from train import train_pretrain_stage, train_sft_stage

train_dataset = build_viet_cultural_dataset("train", n_samples=20, image_size=(256, 256))
val_dataset = build_viet_cultural_dataset("validation", n_samples=8, image_size=(256, 256))

model = build_qlora_model()

train_pretrain_stage(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    output_dir="checkpoints/pretrain_smoke",
    epochs=1,
    batch_size=1,
    grad_accum_steps=1,
    max_steps=1,
    eval_max_steps=1,
)

train_sft_stage(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    output_dir="checkpoints/sft_smoke",
    epochs=1,
    batch_size=1,
    grad_accum_steps=1,
    max_steps=1,
    eval_max_steps=1,
)
```

## Train: Two-Stage VL-JEPA Setup

Stage 1 is query-free pretraining. It uses caption-like targets built from:

```text
answer + detailed_explanation + cultural_significance
```

Stage 2 is query-conditioned SFT. It uses:

```text
question -> answer + detailed_explanation
```

```python
from train import train_pretrain_stage, train_sft_stage

pretrain_history = train_pretrain_stage(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    output_dir="checkpoints/pretrain",
    epochs=1,
    batch_size=2,
    grad_accum_steps=8,
    base_lr=5e-5,
)

sft_history = train_sft_stage(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    output_dir="checkpoints/sft",
    epochs=1,
    batch_size=2,
    grad_accum_steps=8,
    base_lr=5e-5,
)
```

You can also run both stages through the pipeline helper:

```python
from train import run_training_pipeline

histories = run_training_pipeline(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    run_pretrain=True,
    run_sft=True,
    run_one_step=False,
    pretrain_kwargs={
        "output_dir": "checkpoints/pretrain",
        "epochs": 1,
        "batch_size": 2,
        "grad_accum_steps": 8,
    },
    sft_kwargs={
        "output_dir": "checkpoints/sft",
        "epochs": 1,
        "batch_size": 2,
        "grad_accum_steps": 8,
    },
)
```

## Train: One-Step Mixed VQA Baseline

This trains directly on question-answer pairs without a separated pretraining stage.

```python
from train import train_one_step_mixed

baseline_history = train_one_step_mixed(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    output_dir="checkpoints/one_step",
    epochs=1,
    batch_size=2,
    grad_accum_steps=8,
    base_lr=5e-5,
)
```

## Checkpoints

Each training mode saves independently:

- `checkpoints/pretrain`
- `checkpoints/sft`
- `checkpoints/one_step`

Inside each output directory:

- `best`: best validation-loss checkpoint when `val_dataset` is provided.
- `latest`: latest epoch checkpoint.
- `epoch_N`: checkpoint for epoch `N`, when `save_every_epoch=True`.
- `training_state.pt`: epoch, global step, metrics, criterion state, optimizer state, and scheduler state.

If the model supports PEFT saving, adapters are saved with `model.save_pretrained(...)`.

## Evaluate

Evaluation lives in `train.py`, so import it from there.

Use `collate_sft` for VQA evaluation and `collate_pretrain` for query-free alignment evaluation.

```python
import torch
from torch.utils.data import DataLoader

from losses import VLJepaLoss
from train import collate_sft, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_sft,
)

criterion = VLJepaLoss().to(device)

metrics = evaluate(
    model=model,
    dataloader=test_loader,
    criterion=criterion,
    device=device,
    stage_name="test",
)

print(metrics)
```

Returned metrics:

- `loss`: total VL-JEPA loss.
- `info_nce`: symmetric contrastive alignment loss.
- `regularization`: L2 embedding regularization.
- `batches`: number of evaluated batches.

## Inference: Query Embedding

For inference, pass `target_text=None`. The model returns one predicted embedding per image/query pair.

```python
import torch
from train import prepare_pixel_values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

sample = test_dataset[0]
pixel_values = sample["image"].unsqueeze(0)
query = sample["question"]

with torch.no_grad():
    pixel_values = prepare_pixel_values(pixel_values, model, device)
    predicted_embedding = model(
        pixel_values=pixel_values,
        queries_text=[query],
        target_text=None,
    )

print(predicted_embedding.shape)
```

This embedding can be used for retrieval, ranking, clustering, or similarity scoring.

## Inference: Candidate Answer Retrieval

To rank candidate answers, compare the predicted image/query embedding with target-text embeddings.

```python
import torch
import torch.nn.functional as F
from train import prepare_pixel_values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

sample = test_dataset[0]
pixel_values = sample["image"].unsqueeze(0)
query = sample["question"]

candidate_answers = [
    "Day la mot le hoi truyen thong.",
    "Day la mot mon an Viet Nam.",
    "Day la mot di tich lich su.",
]

with torch.no_grad():
    pixel_values = prepare_pixel_values(pixel_values, model, device)
    predicted_embedding, candidate_embeddings = model(
        pixel_values=pixel_values,
        queries_text=[query],
        target_text=candidate_answers,
    )

    scores = F.cosine_similarity(
        F.normalize(predicted_embedding, dim=1),
        F.normalize(candidate_embeddings, dim=1),
    )

best_idx = scores.argmax().item()
print(candidate_answers[best_idx], scores[best_idx].item())
```

## Troubleshooting

If `import torch` fails, install PyTorch in the same Python environment that runs your notebook or script.

If Hugging Face downloads fail, check network access and whether the dataset files still exist in `Dangindev/viet-cultural-vqa`.

If BitsAndBytes fails, verify that your CUDA, PyTorch, and BitsAndBytes versions are compatible. QLoRA is intended for GPU use.

If you run out of VRAM, lower `batch_size`, increase `grad_accum_steps`, reduce `n_samples` for testing, or use `max_steps=1` for a quick smoke run.

If evaluation is slow, pass `eval_max_steps` during training or `max_batches` when calling `evaluate(...)`.
