import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from model import VLJepaModel


def get_compute_dtype():
    """
    Use bf16 if GPU supports it, otherwise use fp16.
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_bnb_config():
    """
    BitsAndBytes config for QLoRA 4-bit loading.
    """
    compute_dtype = get_compute_dtype()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    return bnb_config


def build_lora_config(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
):
    """
    LoRA config for VL-JEPA predictor.

    Important:
    - Do NOT use text_proj because your model does not have self.text_proj.
    - Save custom projection heads using modules_to_save.
    """

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=[
            "vision_proj",
            "predictor_head",
            "y_encoder_head",
        ],
    )

    return peft_config


def freeze_y_encoder_body(model):
    """
    Optional safety step.

    The Y-Encoder is the target encoder, so for small datasets it is usually
    better to freeze its body and only train y_encoder_head.
    """

    for name, param in model.named_parameters():
        if "y_encoder" in name and "y_encoder_head" not in name:
            param.requires_grad = False

    return model


def print_trainable_parameters(model):
    """
    Works for both PEFT and normal PyTorch models.
    """

    trainable_params = 0
    total_params = 0

    for _, param in model.named_parameters():
        total_params += param.numel()

        if param.requires_grad:
            trainable_params += param.numel()

    trainable_percent = 100 * trainable_params / total_params

    print(
        f"Trainable params: {trainable_params:,} "
        f"| Total params: {total_params:,} "
        f"| Trainable: {trainable_percent:.4f}%"
    )


def build_qlora_model(
    vjepa_repo="facebook/vjepa2-vitl-fpc64-256",
    qencode_repo="Qwen/Qwen3-0.6B",
    y_encoder_repo="google/embeddinggemma-300m",
    max_query_len=512,
    shared_embed_dim=1536,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    freeze_y_encoder=True,
):
    """
    Build VL-JEPA with QLoRA.

    Trainable parts:
    - LoRA adapters on Qwen-style attention / MLP modules
    - vision_proj
    - predictor_head
    - y_encoder_head

    Frozen parts:
    - V-JEPA vision encoder
    - original 4-bit base model weights
    - optionally Y-Encoder body
    """

    bnb_config = build_bnb_config()

    model = VLJepaModel(
        vjepa_repo=vjepa_repo,
        qencode_repo=qencode_repo,
        y_encoder_repo=y_encoder_repo,
        max_query_len=max_query_len,
        shared_embed_dim=shared_embed_dim,
        quantization_config=bnb_config,
    )

    model = prepare_model_for_kbit_training(model)

    peft_config = build_lora_config(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    model = get_peft_model(model, peft_config)

    if freeze_y_encoder:
        model = freeze_y_encoder_body(model)

    print_trainable_parameters(model)

    return model
