# VL-JEPA Vietnamese Cultural VQA

This project adapts a VL-JEPA-style joint embedding predictive architecture for Vietnamese cultural visual question answering (VQA). The goal is to learn a shared embedding space where an image plus an optional query predicts the embedding of the correct target answer, explanation, or cultural description.

The current system is an embedding and retrieval model. It does not directly generate natural-language answers. During inference, it returns a predicted target embedding that can be compared with candidate answer embeddings using cosine similarity or another retrieval metric.

## External Sources and Attribution

This repository composes pretrained models and a public dataset into a custom training pipeline.

| Component | Source |
| --- | --- |
| Main architecture | [VL-JEPA paper](https://arxiv.org/pdf/2512.10942) |
| Query/predictor backbone | [Qwen3-0.6B paper](https://arxiv.org/pdf/2505.09388) |
| Target text encoder | [google/embeddinggemma-300m on Hugging Face](https://huggingface.co/google/embeddinggemma-300m) |
| Dataset | [Dangindev/viet-cultural-vqa on Hugging Face](https://huggingface.co/datasets/Dangindev/viet-cultural-vqa) |

Default model identifiers used by the code:

- Vision encoder: `facebook/vjepa2-vitl-fpc64-256`
- Query/predictor source model: `Qwen/Qwen3-0.6B`
- Target text encoder: `google/embeddinggemma-300m`
- Dataset repo: `Dangindev/viet-cultural-vqa`

## Architecture Overview

The model follows the VL-JEPA idea of predicting a target representation rather than generating text tokens. It has two embedding paths:

- The predictor path receives visual input and a query, then predicts the target text embedding.
- The target path encodes the real answer/explanation text into the same shared embedding space.

Training aligns these two embeddings with contrastive learning.

```text
Image
  -> V-JEPA vision encoder frozen
  -> vision_proj trainable
  -> vision tokens in Qwen hidden size
                                  \
                                   -> concatenate -> Qwen predictor last 8 layers -> mean pool query tokens -> predictor_head -> S_Y_hat
                                  /
Question / query
  -> Qwen tokenizer
  -> Qwen token embedding

Target answer / explanation
  -> EmbeddingGemma tokenizer
  -> EmbeddingGemma encoder
  -> masked mean pooling
  -> y_encoder_head
  -> S_Y

Loss: InfoNCE for multi-sample batches, cosine alignment for singleton batches, plus L2 regularization
```

Important dimensions:

- V-JEPA hidden size is read from the V-JEPA config at runtime.
- Qwen hidden size is read from the Qwen config at runtime.
- `vision_proj` maps V-JEPA hidden states into Qwen hidden size.
- `predictor_head` and `y_encoder_head` map both branches into a shared embedding space, currently `shared_embed_dim=1536`.

## Architecture in Detail

### X-Encoder / Vision Encoder

The X-encoder uses `facebook/vjepa2-vitl-fpc64-256`. It is frozen by default, so gradients do not update the V-JEPA vision backbone during training.

The dataset provides image tensors as `[B, C, H, W]`. The model adapts these images to the V-JEPA video-style interface by inserting a frame dimension and repeating the image across frames. This lets an image-only sample pass through the video-oriented V-JEPA encoder.

The V-JEPA encoder outputs visual hidden states. These visual states are the image-side representation before projection.

### Vision Projection

`vision_proj` is a trainable linear layer that maps V-JEPA hidden states into Qwen hidden size. This is the bridge between the frozen vision encoder and the Qwen predictor stack.

Because this layer is specific to this project, it is included in `modules_to_save` so it is saved with the PEFT adapter checkpoint.

### Query Text Embedder

The query path uses the Qwen3 tokenizer and token embedding from `Qwen/Qwen3-0.6B`. The query can be:

- empty or neutral during query-free pretraining,
- a VQA question during supervised finetuning,
- a user question during retrieval-style inference.

The token embeddings live in the same hidden size expected by the Qwen predictor layers.

### Predictor

The predictor uses the last 8 transformer layers and final norm from Qwen3-0.6B. It receives a concatenated sequence:

```text
[vision_tokens, query_tokens]
```

The model builds a bidirectional attention mask so visual tokens and query tokens can attend to each other. This is different from normal causal language modeling: the predictor is used as a bidirectional representation module for alignment, not as an autoregressive decoder.

After the predictor layers, the model takes the query-token output positions, applies masked mean pooling over non-padding tokens, and sends the pooled vector through `predictor_head`. The result is:

```text
S_Y_hat
```

`S_Y_hat` is the predicted embedding of the target answer/explanation.

### Y-Encoder / Target Encoder

The Y-encoder uses `google/embeddinggemma-300m`. It encodes the target text, such as an answer, detailed explanation, or cultural significance text.

The target hidden states are mean-pooled over non-padding tokens. Then `y_encoder_head` projects the pooled target representation into the same shared embedding dimension as the predictor output:

```text
S_Y
```

By default, the Y-encoder body is frozen for small-dataset stability, while `y_encoder_head` remains trainable.

### Loss

Training uses `VLJepaLoss`, which combines:

- symmetric InfoNCE contrastive loss when the actual batch has at least two samples,
- in-batch negatives for multi-sample batches,
- cosine alignment fallback when the actual batch has one sample,
- a learnable temperature by default,
- L2 regularization on predicted and target embeddings.

For each batch, the positive pair is on the diagonal of the similarity matrix:

```text
predicted embedding i <-> target embedding i
```

Other batch items become negatives.

If the DataLoader produces a singleton batch, such as a one-sample dataset or a final leftover sample, the loss skips InfoNCE because there are no negatives and uses `1 - cosine_similarity(predicted, target)` instead. This keeps small smoke tests and low-VRAM runs trainable, though larger batches still provide a stronger contrastive signal.

## QLoRA Setup

The project uses QLoRA to make finetuning practical with large pretrained components.

BitsAndBytes 4-bit loading is configured with:

- `load_in_4bit=True`
- NF4 quantization
- double quantization
- bf16 compute when the GPU supports it, otherwise fp16

LoRA adapters are applied to Qwen-style attention and MLP projection modules:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

Additional trainable project-specific modules are saved with the adapter:

- `vision_proj`
- `predictor_head`
- `y_encoder_head`

Frozen by default:

- V-JEPA vision encoder
- base 4-bit pretrained weights
- Y-encoder body, unless configured otherwise

QLoRA is useful here because it reduces memory use while still allowing the predictor and projection layers to adapt to the Vietnamese cultural VQA task.

## Dataset Pipeline

The dataset is loaded from [Dangindev/viet-cultural-vqa](https://huggingface.co/datasets/Dangindev/viet-cultural-vqa).

Supported split files:

- `splits/train_data.json`
- `splits/val_data.json`
- `splits/test_data.json`

The loader pipeline:

1. Scans files in the Hugging Face dataset repo.
2. Downloads the selected split JSON.
3. Resolves each image path.
4. Downloads each image from Hugging Face.
5. Converts images to RGB NumPy arrays.
6. Skips samples whose images fail to download.
7. Expands each image-level record into one PyTorch sample per QA pair.

Each dataset item includes:

- `image`
- `question`
- `answer`
- `detailed_explanation`
- `cultural_significance`
- `additional_context`
- `difficulty`
- `question_type`
- `cognitive_level`
- identifiers and image path metadata

## Training Pipeline

The repository supports four main run modes.

### Smoke

Smoke mode runs a tiny 1-step pretraining pass and a tiny 1-step SFT pass. It is intended for checking that the environment, dataset loading, model loading, and training loop are wired correctly.

### Stage 1: Query-Free Pretraining

Stage 1 uses `collate_pretrain`.

The query is empty or neutral. The target text is built from:

```text
answer + detailed_explanation + cultural_significance
```

The goal is to build image-to-language embedding alignment before the model becomes strongly conditioned on VQA questions. The default scheduler for this stage is constant learning rate.

### Stage 2: Query-Conditioned SFT

Stage 2 uses `collate_sft`.

The query is the VQA question. The target text is built from:

```text
answer + detailed_explanation
```

The goal is to make the predictor produce target embeddings conditioned on the question, not just the image. The default scheduler for this stage is cosine learning rate annealing.

### One-Step Mixed VQA Baseline

The one-step baseline trains directly with the SFT collator. It skips the separated pretraining stage and directly optimizes question-conditioned VQA alignment.

This is useful for fast experiments and as a comparison against the two-stage VL-JEPA-style pipeline.

## Evaluation and Inference

Evaluation uses the same contrastive objective as training and reports:

- `loss`
- `info_nce`
- `align_loss`
- `regularization`
- `batches`

Inference is embedding-based:

- pass `target_text=None` to get a predicted embedding for an image/query pair,
- encode candidate target answers,
- compare predicted and candidate embeddings with cosine similarity,
- choose or rank the highest-scoring candidates.

This means the current model is suitable for retrieval, ranking, and representation learning. It is not yet a text-generating VQA assistant.

## Quick Start

From the project root:

```bash
bash scripts/run_smoke.sh
bash scripts/run_two_stage.sh
bash scripts/run_one_step.sh
```

Direct CLI examples:

```bash
python src/train.py --scenario smoke
python src/train.py --scenario two_stage --train-samples 100 --batch-size 1
```

All omitted CLI and script config values use defaults. The default `batch_size` is `1` so scripts can run in low-VRAM or tiny-sample settings; use `BATCH_SIZE=2` or higher when possible for stronger InfoNCE negatives. For complete CLI, notebook, evaluation, and inference examples, see [src/README.md](src/README.md). For the concise training-stage summary, see [src/TRAINING_STRUCTURE.md](src/TRAINING_STRUCTURE.md).

## Repository Structure

```text
VL-JEPA/
  README.md
  requirements.txt
  scripts/
    run_smoke.sh
    run_pretrain.sh
    run_sft.sh
    run_two_stage.sh
    run_one_step.sh
    run_eval_validation.sh
    run_eval_test.sh
    run_all.sh
  src/
    model.py
    QLoRA_setup.py
    dataset.py
    train.py
    losses.py
    README.md
    TRAINING_STRUCTURE.md
```

## Outputs

Training writes checkpoints under `checkpoints/`.

Common scenario folders:

- `checkpoints/smoke`
- `checkpoints/pretrain`
- `checkpoints/sft`
- `checkpoints/one_step`

Each training output may contain:

- `best`: best validation-loss checkpoint when validation is available,
- `latest`: latest checkpoint,
- `epoch_N`: checkpoint for a specific epoch,
- `training_state.pt`: optimizer, scheduler, criterion, epoch, global step, and metrics.

## Limitations

- The current model returns embeddings, not generated text answers.
- CLI evaluation currently builds a model and evaluates it; checkpoint loading for CLI evaluation is not implemented yet.
- Dataset image loading depends on Hugging Face availability and network access.
- QLoRA/BitsAndBytes is intended for CUDA GPU environments.
- The training setup is designed for experimentation and may need scaling, checkpoint resume logic, and stronger validation before production use.
