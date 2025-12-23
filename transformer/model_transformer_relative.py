import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 基础组件：RMSNorm & RoPE
# ==========================================

class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)
    带有 bias 参数以通过 PyTorch 内部检查
    """
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = d_model ** -0.5
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.register_parameter("bias", nn.Parameter(torch.zeros(d_model)))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    """
    RoPE: Rotary Positional Embedding
    """
    def __init__(self, dim, max_seq_len=5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        # freqs: (seq_len, dim/2)
        # 我们需要生成 sin, cos 用于旋转
        emb = torch.cat((freqs, freqs), dim=-1) # (seq_len, dim)
        self.register_buffer('cos', emb.cos())
        self.register_buffer('sin', emb.sin())

    def forward(self, x, seq_len=None):
        # x: (Batch, Seq, Dim) or (Batch, Head, Seq, Head_Dim)
        # 返回对应的 cos, sin
        if seq_len is None:
            seq_len = x.shape[1]
        return self.cos[:seq_len, :], self.sin[:seq_len, :]

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    q, k: (Batch, Heads, Seq_Len, Head_Dim)
    cos, sin: (Seq_Len, Head_Dim) -> 需要 reshape 广播
    """
    # 调整 cos, sin 形状以匹配 (1, 1, Seq_Len, Head_Dim)
    cos = cos.unsqueeze(0).unsqueeze(0) 
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ==========================================
# 2. 自定义 Attention (支持 RoPE)
# ==========================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, max_len=5000):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_len)

    def forward(self, x, key_padding_mask=None, attn_mask=None, is_cross_attn=False, context=None):
        """
        x: (Batch, Seq_Len, D_Model)
        context: 用于 Cross Attention 的 Encoder Output
        """
        B, S, D = x.shape
        
        # 1. 投影 Q
        q = self.q_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2) # (B, H, S, D_h)
        
        # 2. 投影 K, V
        if is_cross_attn and context is not None:
            # Cross Attention: K, V 来自 Encoder Output (context)
            S_k = context.shape[1]
            k = self.k_proj(context).view(B, S_k, self.nhead, self.head_dim).transpose(1, 2)
            v = self.v_proj(context).view(B, S_k, self.nhead, self.head_dim).transpose(1, 2)
            # Cross Attention 通常不加 RoPE，或者只加在 Query 上（这里为了简单，Cross Attn 不加 RoPE）
        else:
            # Self Attention
            k = self.k_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
            
            # Apply RoPE (仅在 Self Attention 中使用)
            cos, sin = self.rope(x, seq_len=S)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. 计算 Attention
        # PyTorch 2.0+ 推荐使用 scaled_dot_product_attention (自动处理 mask 和 dropout)
        # attn_mask 处理: (S, S) or (T, S)
        
        # 处理 padding mask (B, S) -> 扩展形状
        # 注意：SDPA 接受的 mask 逻辑可能需要调整，这里用手动实现最稳妥兼容
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, H, S, S_k)
        
        if attn_mask is not None:
            # attn_mask: (S, S) -> 广播到 (B, H, S, S)
            # Mask 为 -inf 的位置
            scores = scores + attn_mask 
            
        if key_padding_mask is not None:
            # key_padding_mask: (B, S_k) True 为 pad
            # 需要扩展为 (B, 1, 1, S_k) 并广播
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1) # (B, 1, 1, S_k)
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v) # (B, H, S, D_h)
        
        # 4. 拼接 Heads
        output = output.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(output)

# ==========================================
# 3. Encoder / Decoder Layers
# ==========================================

class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_type):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, nhead, dropout)
        
        # Norm
        NormClass = RMSNorm if norm_type == 'rms' else nn.LayerNorm
        self.norm1 = NormClass(d_model)
        self.norm2 = NormClass(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-Norm 结构 (更稳定)
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, key_padding_mask=src_key_padding_mask, attn_mask=None) # Encoder 自注意力通常不需要 causal mask
        src = src + self.dropout(src2)
        
        src2 = self.norm2(src)
        src2 = self.ffn(src2)
        src = src + src2
        return src

class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_type):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, nhead, dropout)
        self.cross_attn = CausalSelfAttention(d_model, nhead, dropout)
        
        NormClass = RMSNorm if norm_type == 'rms' else nn.LayerNorm
        self.norm1 = NormClass(d_model)
        self.norm2 = NormClass(d_model)
        self.norm3 = NormClass(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 1. Self Attention (Masked)
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        
        # 2. Cross Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(tgt2, key_padding_mask=memory_key_padding_mask, is_cross_attn=True, context=memory)
        tgt = tgt + self.dropout(tgt2)
        
        # 3. FFN
        tgt2 = self.norm3(tgt)
        tgt2 = self.ffn(tgt2)
        tgt = tgt + tgt2
        return tgt

# ==========================================
# 4. 主模型 (API 兼容 train.py)
# ==========================================

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3, 
                 dim_feedforward=512, dropout=0.1, norm_type='layer'):
        super().__init__()
        
        self.d_model = d_model
        print(f"Initializing RoPE Transformer (Relative Position) with {norm_type.upper()} Norm")

        # Embeddings (No Absolute Positional Encoding Needed!)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            CustomEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_type)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            CustomDecoderLayer(d_model, nhead, dim_feedforward, dropout, norm_type)
            for _ in range(num_decoder_layers)
        ])
        
        self.out = nn.Linear(d_model, tgt_vocab_size)
        
        # 兼容性接口：train.py 可能会调用 model.transformer_encoder
        # 我们用一个简单的 lambda 或者 wrapper 来模拟
        self.transformer_encoder = self.forward_encoder
        self.transformer_decoder = self.forward_decoder
        self.pos_encoder = lambda x: self.dropout(x) # 只是 dropout，不加绝对位置编码

    def forward_encoder(self, src, src_key_padding_mask=None):
        output = src
        for layer in self.encoder_layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)
        return output

    def forward_decoder(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask=tgt_mask, 
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        return output

    def forward(self, src, tgt, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        # src: (B, S), tgt: (B, T)
        
        # 1. Embedding (No Abs Pos)
        src_emb = self.dropout(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.dropout(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        # 2. Encoder
        memory = self.forward_encoder(src_emb, src_key_padding_mask=src_padding_mask)

        # 3. Decoder
        output = self.forward_decoder(tgt_emb, memory, tgt_mask=tgt_mask, 
                                      tgt_key_padding_mask=tgt_padding_mask,
                                      memory_key_padding_mask=src_padding_mask)
        
        return self.out(output)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask