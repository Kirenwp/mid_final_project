import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 对应要求：Two unidirectional layers
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, 
                          batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, Seq)
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: (1, B, H) -> 取最后一层 (B, H)
        # encoder_outputs: (B, Seq, H)
        
        # 统一把 hidden 转为 (B, H)
        if hidden.dim() == 3:
            hidden = hidden[-1]

        seq_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)

        # 计算 Attention Energies
        attn_energies = torch.zeros(batch_size, seq_len).to(encoder_outputs.device) 

        if self.method == 'dot':
            # (B, 1, H) * (B, H, Seq) -> (B, 1, Seq)
            attn_energies = torch.bmm(hidden.unsqueeze(1), encoder_outputs.transpose(1, 2)).squeeze(1)
            
        elif self.method == 'general':
            # General: hidden * W * encoder_outputs
            attn_layer = self.attn(encoder_outputs) # (B, Seq, H)
            attn_energies = torch.bmm(hidden.unsqueeze(1), attn_layer.transpose(1, 2)).squeeze(1)
            
        elif self.method == 'concat':
            # Additive: v * tanh(W * [hidden; encoder_outputs])
            # 扩展 hidden 以匹配 seq_len: (B, Seq, H)
            hidden_expanded = hidden.unsqueeze(1).expand(-1, seq_len, -1)
            concat_input = torch.cat((hidden_expanded, encoder_outputs), 2) # (B, Seq, 2H)
            energy = torch.tanh(self.attn(concat_input)) # (B, Seq, H)
            
            # 与 v 点积
            # v: (H) -> (1, 1, H) -> (B, 1, H)
            v = self.v.repeat(batch_size, 1, 1)
            attn_energies = torch.bmm(v, energy.transpose(1, 2)).squeeze(1) # (B, Seq)

        return F.softmax(attn_energies, dim=1).unsqueeze(1) # (B, 1, Seq)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=2, dropout=0.1, attn_method='dot'):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, batch_first=True)
        
        # 实例化 Attention 模块
        self.attn = Attention(attn_method, hidden_size)
        
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, hidden, encoder_outputs):
        # input: (B, 1)
        embedded = self.dropout(self.embedding(input)) # (B, 1, H)
        
        # GRU 步进
        rnn_output, hidden = self.gru(embedded, hidden) # rnn_output: (B, 1, H)
        
        # 计算 Attention权重
        attn_weights = self.attn(rnn_output, encoder_outputs) # (B, 1, Seq)
        
        # 加权求和上下文
        context = torch.bmm(attn_weights, encoder_outputs) # (B, 1, H)
        
        # 拼接 RNN 输出和 Context
        output = torch.cat((rnn_output, context), dim=2) # (B, 1, 2H)
        
        output = self.fc(output) # (B, 1, Vocab)
        
        return output, hidden, attn_weights