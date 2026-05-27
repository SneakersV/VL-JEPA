import torch
import torch.nn as nn
from transformers import AutoVideoProcessor, AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class VLJepaModel(nn.Module):
    def __init__(
        self, 
        vjepa_repo="facebook/vjepa2-vitl-fpc64-256", 
        qencode_repo="Qwen/Qwen3-0.6B",
        y_encoder_repo="google/embeddinggemma-300m",
        max_query_len=512,
        shared_embed_dim=1536,
        quantization_config=None
    ):
        super().__init__()
        self.max_query_len = max_query_len
        
        # ==========================================
        # 1. X-ENCODER (FROZEN VISION MODEL)
        # ==========================================
        self.vision_processor = AutoVideoProcessor.from_pretrained(vjepa_repo)
        self.vision_encoder = AutoModel.from_pretrained(
            vjepa_repo,
            quantization_config=quantization_config)
        
        # Freeze V-JEPA 2
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # ==========================================
        # 2. TEXT ENCODER & TOKENIZER (Từ Qwen3-0.6B)
        # ==========================================
        self.tokenizer = AutoTokenizer.from_pretrained(qencode_repo)

        # Qwen mặc định có thể không thiết lập sẵn PAD token, ta mượn tạm EOS token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        qwen_full = AutoModel.from_pretrained(
            qencode_repo,
            quantization_config=quantization_config)

        # Lấy riêng Token Embedding
        self.text_embedder = qwen_full.embed_tokens
        
        # ==========================================
        # 3. PREDICTOR (8 LỚP QWEN CUỐI CÙNG)
        # ==========================================
        # Cắt lấy 8 lớp Transformer cuối cùng của Qwen
        self.predictor_layers = nn.ModuleList(qwen_full.layers[-8:])
        self.predictor_norm = qwen_full.norm
        self.qwen_rotary_emb = qwen_full.rotary_emb
        self.qwen_config = qwen_full.config
        
        # Linear projections
        vision_dim = self.vision_encoder.config.hidden_size
        qwen_dim = qwen_full.config.hidden_size

        self.vision_proj = nn.Linear(vision_dim, qwen_dim)
        self.predictor_head = nn.Linear(qwen_dim, shared_embed_dim)

        # Giải phóng bộ nhớ model Qwen gốc
        del qwen_full
        
        # ==========================================
        # 4. Y-ENCODER (EmbeddingGemma-300M)
        # ==========================================
        self.y_tokenizer = AutoTokenizer.from_pretrained(y_encoder_repo)
        if self.y_tokenizer.pad_token is None:
            self.y_tokenizer.pad_token = self.y_tokenizer.eos_token
            
        self.y_encoder = AutoModel.from_pretrained(
            y_encoder_repo,
            quantization_config=quantization_config
        )
        
        # Final projection head for the Y-Encoder to reach 1,536 dimensions
        gemma_dim = self.y_encoder.config.hidden_size
        self.y_encoder_head = nn.Linear(gemma_dim, shared_embed_dim)


    def encode_vision(self, pixel_values):
        """Convert image/video tensors to V-JEPA's video input layout."""
        # V-JEPA2's HF forward expects [B, T, C, H, W] and permutes internally
        # before its Conv3d patch embedding. For still images, repeat one frame.
        if len(pixel_values.shape) == 4:
            pixel_values = pixel_values.unsqueeze(1).repeat(1, 16, 1, 1, 1)
        elif len(pixel_values.shape) == 5:
            if pixel_values.shape[2] == 3:
                pass
            elif pixel_values.shape[1] == 3:
                pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
            else:
                raise ValueError(
                    "Expected video pixel_values in [B, T, C, H, W] or "
                    f"[B, C, T, H, W] format, got shape {tuple(pixel_values.shape)}."
                )
        else:
            raise ValueError(
                "Expected image pixel_values [B, C, H, W] or video pixel_values "
                f"[B, T, C, H, W], got shape {tuple(pixel_values.shape)}."
            )

        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values_videos=pixel_values)
            vision_features = vision_outputs.last_hidden_state 
            
        return self.vision_proj(vision_features)


    def _create_bidirectional_mask(self, vision_len, text_mask, dtype, device):
        """
        Tạo Custom Attention Mask để disable Causal Mask.
        Biến Llama/Qwen thành mạng 2 chiều (Bidirectional) cho phép vision và text nhìn thấy nhau.
        """
        batch_size, text_len = text_mask.shape
        seq_len = vision_len + text_len
        
        # Tạo mask 2D toàn số 1 (cho phép attention)
        # Vision tokens luôn là 1 (không có pad). Text tokens dùng text_mask (có pad).
        vision_mask = torch.ones((batch_size, vision_len), dtype=torch.long, device=device)
        joint_mask_2d = torch.cat([vision_mask, text_mask], dim=1)
        
        # Chuyển đổi sang định dạng 4D mask của Hugging Face: [Batch, 1, Seq, Seq]
        # Chỗ nào giá trị là 0 (PAD) thì chuyển thành số âm vô cùng để triệt tiêu attention
        extended_mask = joint_mask_2d[:, None, None, :]
        extended_mask = extended_mask.to(dtype=dtype)
        extended_mask = (1.0 - extended_mask) * torch.finfo(dtype).min
        
        # Trả về mask vuông [B, 1, S, S] thay vì ma trận tam giác (causal mask)
        return extended_mask.expand(-1, -1, seq_len, -1)

    def forward(self, pixel_values, queries_text=None, target_text=None):
        """
        Args:
            pixel_values: Vision input tensors.
            queries_text: Textual query X_Q (e.g., "Describe this video").
            target_text: Textual target Y (e.g., the actual caption). Required during training.
        """
        device = pixel_values.device
        batch_size = pixel_values.shape[0]
        
        # 1. Trích xuất đặc trưng Vision
        vision_embeds = self.encode_vision(pixel_values) # Shape: [B, N_vision, D]
        vision_len = vision_embeds.shape[1]
        
        # 2. Xử lý Text Queries (Padding & Truncation)
        if queries_text is None:
            queries_text = [""] * batch_size
        elif isinstance(queries_text, str):
            queries_text = [queries_text] * batch_size
            
        tokens = self.tokenizer(
            queries_text, 
            padding=True, 
            truncation=True, 
            max_length=self.max_query_len, 
            return_tensors="pt"
        ).to(device)
        
        # Lấy Text Embeddings
        text_embeds = self.text_embedder(tokens.input_ids)
        
        # 3. Ghép nối chuỗi (Joint Embedding)
        hidden_states = torch.cat([vision_embeds, text_embeds], dim=1)
        batch_size, seq_len, _ = hidden_states.shape
        
        # 4. Tạo Bidirectional Mask (Tắt Causal Mask)
        attention_mask = self._create_bidirectional_mask(
            vision_len=vision_len,
            text_mask=tokens.attention_mask,
            dtype=hidden_states.dtype,
            device=device
        )
        
        position_ids = torch.arange(
            0, seq_len, dtype=torch.long, device=hidden_states.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        position_embeddings = self.qwen_rotary_emb(hidden_states, position_ids)

        # 5. Đưa qua 8 lớp Predictor (Llama layers)
        for layer in self.predictor_layers:
            layer_outputs = layer(
                hidden_states, 
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            
        hidden_states = self.predictor_norm(hidden_states)
        
        # 6. Average Pooling trên Non-[PAD] text tokens
        # Tách phần output thuộc về Text
        text_outputs = hidden_states[:, vision_len:, :]
        
        # Dùng attention_mask (1 cho chữ thực, 0 cho PAD) để lọc
        mask = tokens.attention_mask.unsqueeze(-1).to(text_outputs.dtype)
        
        # Tính tổng các vector hợp lệ chia cho số lượng token hợp lệ
        sum_embeddings = (text_outputs * mask).sum(dim=1)
        valid_token_counts = mask.sum(dim=1).clamp(min=1e-9)
        
        predicted_target_embedding = sum_embeddings / valid_token_counts
        predicted_target_embedding = self.predictor_head(predicted_target_embedding)
        
        if target_text is None:
            return predicted_target_embedding
        
        # --- PATH 2: Y-ENCODER PIPELINE (Generates S_Y) ---
        if isinstance(target_text, str):
            target_text = [target_text] * batch_size
            
        y_tokens = self.y_tokenizer(
            target_text, 
            padding=True, 
            truncation=True, 
            max_length=self.max_query_len,
            return_tensors="pt"
        ).to(device)
        
        y_outputs = self.y_encoder(**y_tokens)
        y_hidden_states = y_outputs.last_hidden_state
        
        y_mask = y_tokens.attention_mask.unsqueeze(-1).to(y_hidden_states.dtype)
        y_sum_embeddings = (y_hidden_states * y_mask).sum(dim=1)
        y_valid_token_counts = y_mask.sum(dim=1).clamp(min=1e-9)
        
        pooled_target_embedding = y_sum_embeddings / y_valid_token_counts
        target_embedding = self.y_encoder_head(pooled_target_embedding)
        
        return predicted_target_embedding, target_embedding
