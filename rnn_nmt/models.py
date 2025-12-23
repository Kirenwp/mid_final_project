\
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float, rnn_type: str = "gru"):
        super().__init__()
        assert rnn_type in ("gru", "lstm")
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(emb_dim, hid_dim, num_layers=n_layers, bidirectional=False, dropout=dropout if n_layers > 1 else 0.0)

    def forward(self, src: torch.Tensor):
        # src: [src_len, batch]
        embedded = self.dropout(self.embedding(src))  # [src_len, batch, emb_dim]
        outputs, hidden = self.rnn(embedded)          # outputs: [src_len, batch, hid_dim]
        return outputs, hidden  # hidden: [n_layers, batch, hid_dim] (or tuple for LSTM)

class Attention(nn.Module):
    """
    Alignment functions:
      - dot: score(h_t, h_s) = h_t^T h_s
      - general (multiplicative): score = h_t^T W h_s  (Luong general)
      - additive: score = v^T tanh(W_s h_s + W_t h_t)  (Bahdanau)
    """
    def __init__(self, hid_dim: int, attn_type: str = "dot"):
        super().__init__()
        assert attn_type in ("dot", "general", "additive")
        self.attn_type = attn_type
        self.hid_dim = hid_dim
        if attn_type == "general":
            self.W = nn.Linear(hid_dim, hid_dim, bias=False)
        elif attn_type == "additive":
            self.W_s = nn.Linear(hid_dim, hid_dim, bias=False)
            self.W_t = nn.Linear(hid_dim, hid_dim, bias=False)
            self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, dec_hidden_last: torch.Tensor, enc_outputs: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        """
        dec_hidden_last: [batch, hid_dim]  (decoder last layer hidden)
        enc_outputs:     [src_len, batch, hid_dim]
        src_mask:        [batch, src_len]  True for real tokens, False for PAD
        returns attention weights: [batch, src_len]
        """
        src_len, batch, hid_dim = enc_outputs.shape
        enc = enc_outputs.permute(1, 0, 2)  # [batch, src_len, hid_dim]

        if self.attn_type == "dot":
            # [batch, src_len]
            energy = torch.bmm(enc, dec_hidden_last.unsqueeze(2)).squeeze(2)
        elif self.attn_type == "general":
            enc_w = self.W(enc)  # [batch, src_len, hid_dim]
            energy = torch.bmm(enc_w, dec_hidden_last.unsqueeze(2)).squeeze(2)
        else:  # additive
            # expand dec hidden to all time steps
            dec = dec_hidden_last.unsqueeze(1).expand(-1, src_len, -1)  # [batch, src_len, hid_dim]
            energy = self.v(torch.tanh(self.W_s(enc) + self.W_t(dec))).squeeze(2)  # [batch, src_len]

        if src_mask is not None:
            energy = energy.masked_fill(~src_mask, float("-inf"))

        attn = F.softmax(energy, dim=1)
        return attn

class Decoder(nn.Module):
    def __init__(self, output_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float,
                 attention: Attention, rnn_type: str = "gru"):
        super().__init__()
        assert rnn_type in ("gru", "lstm")
        self.rnn_type = rnn_type
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        # input is [emb_dim + hid_dim] because we concat context
        self.rnn = rnn_cls(emb_dim + hid_dim, hid_dim, num_layers=n_layers, bidirectional=False, dropout=dropout if n_layers > 1 else 0.0)

        self.fc_out = nn.Linear(hid_dim + hid_dim + emb_dim, output_dim)

    def _last_hidden(self, hidden):
        # hidden: [n_layers, batch, hid_dim] or (h, c)
        if self.rnn_type == "lstm":
            h, c = hidden
            return h[-1]  # [batch, hid_dim]
        return hidden[-1]

    def forward(self, input_tok: torch.Tensor, hidden, enc_outputs: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        """
        input_tok: [batch] (token ids)
        hidden: [n_layers, batch, hid_dim] or tuple for LSTM
        enc_outputs: [src_len, batch, hid_dim]
        src_mask: [batch, src_len]
        """
        input_tok = input_tok.unsqueeze(0)  # [1, batch]
        embedded = self.dropout(self.embedding(input_tok))  # [1, batch, emb_dim]

        dec_hidden_last = self._last_hidden(hidden)  # [batch, hid_dim]
        attn_weights = self.attention(dec_hidden_last, enc_outputs, src_mask=src_mask)  # [batch, src_len]

        enc = enc_outputs.permute(1, 0, 2)  # [batch, src_len, hid_dim]
        context = torch.bmm(attn_weights.unsqueeze(1), enc).permute(1, 0, 2)  # [1, batch, hid_dim]

        rnn_input = torch.cat((embedded, context), dim=2)  # [1, batch, emb_dim+hid_dim]
        output, hidden = self.rnn(rnn_input, hidden)       # output: [1, batch, hid_dim]
        output = output.squeeze(0)
        context = context.squeeze(0)
        embedded = embedded.squeeze(0)
        pred = self.fc_out(torch.cat((output, context, embedded), dim=1))  # [batch, output_dim]
        return pred, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, pad_idx: int, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def make_src_mask(self, src: torch.Tensor):
        # src: [src_len, batch]
        # returns [batch, src_len]
        return (src.transpose(0,1) != self.pad_idx)

    def forward(self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        """
        src: [src_len, batch]
        trg: [trg_len, batch]
        returns outputs: [trg_len, batch, output_dim]
        """
        trg_len, batch = trg.shape
        output_dim = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch, output_dim, device=self.device)

        enc_outputs, hidden = self.encoder(src)
        src_mask = self.make_src_mask(src)

        input_tok = trg[0]  # first is <sos>
        for t in range(1, trg_len):
            pred, hidden, _ = self.decoder(input_tok, hidden, enc_outputs, src_mask=src_mask)
            outputs[t] = pred
            teacher_force = (torch.rand(1).item() < teacher_forcing_ratio)
            top1 = pred.argmax(1)
            input_tok = trg[t] if teacher_force else top1

        return outputs
