import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    """
    作业要求的 RMSNorm 实现 (Root Mean Square Layer Normalization)
    
    修正说明：
    1. 将参数命名为 'weight' 以兼容 PyTorch 的检查。
    2. [关键] 添加 'bias' 参数并初始化为 0。
       虽然数学上 RMSNorm 不需要 bias，但 PyTorch 的 nn.Transformer 组件
       在底层优化时会强制访问 .bias 属性。如果不加这个，会报 AttributeError。
    """
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = d_model ** -0.5
        self.eps = eps
        # 1. 必须叫 weight，不能叫 g
        self.weight = nn.Parameter(torch.ones(d_model))
        # 2. 必须有 bias，即使我们不用它。初始化为 0 以保证数学等价性。
        self.register_parameter("bias", nn.Parameter(torch.zeros(d_model)))

    def forward(self, x):
        # 计算 RMS
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        # 标准公式：x / RMS * weight
        # 注意：我们这里不需要加 self.bias，因为 RMSNorm 本身没有位移。
        # 这里的 self.bias 只是为了占位，防止 PyTorch 报错。
        return x / (norm + self.eps) * self.weight

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (Batch, Seq, Feature)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3, 
                 dim_feedforward=512, dropout=0.1, norm_type='layer'):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.norm_type = norm_type

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 定义 Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)

        # 核心修改：如果是 RMSNorm，替换内部的 LayerNorm
        if norm_type == 'rms':
            print("Applying RMSNorm (Replacing LayerNorm with Custom RMSNorm)")
            # 替换 Encoder 里的 Norm
            encoder_layer.norm1 = RMSNorm(d_model)
            encoder_layer.norm2 = RMSNorm(d_model)
            # 替换 Decoder 里的 Norm
            decoder_layer.norm1 = RMSNorm(d_model)
            decoder_layer.norm2 = RMSNorm(d_model)
            decoder_layer.norm3 = RMSNorm(d_model)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        # src: (B, S), tgt: (B, T)
        
        # 1. Embedding + Positional Encoding
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        # 2. Encoder
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)

        # 3. Decoder
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask, 
                                          tgt_key_padding_mask=tgt_padding_mask,
                                          memory_key_padding_mask=src_padding_mask)
        
        # 4. Output
        return self.out(output)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask