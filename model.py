import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM_CRF(nn.Module):
    """
    Bi-directional LSTM + CRF cho NER.
    Sử dụng pytorch-crf mà không dùng batch_first khi khởi tạo CRF.
    """
    def __init__(
        self,
        vocab_size: int,
        tagset_size: int,
        emb_dim: int = 100,
        hid_dim: int = 128,
        dropout: float = 0.5,
        padding_idx: int = 0
    ):
        super().__init__()
        # Embedding + Dropout
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.emb_dropout = nn.Dropout(dropout)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.lstm_dropout = nn.Dropout(dropout)

        # Linear to get emission scores
        self.fc = nn.Linear(hid_dim, tagset_size)

        # CRF layer (pytorch-crf) without batch_first arg
        self.crf = CRF(tagset_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].zero_()
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x_ids, tags=None, mask=None, lengths=None):
        # Embedding + dropout
        emb = self.embedding(x_ids)              # [B, L, emb_dim]
        emb = self.emb_dropout(emb)

        # LSTM (packed if lengths given)
        if lengths is not None:
            packed = pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(emb)         # [B, L, hid_dim]
        lstm_out = self.lstm_dropout(lstm_out)

        # Emissions scores
        emissions = self.fc(lstm_out)            # [B, L, tagset]

        if tags is not None:
            # CRF forward returns log-likelihood; negate for loss
            log_likelihood = self.crf(emissions, tags, mask=mask)
            return -log_likelihood.mean()
        # Decode with CRF.viterbi_tags or fallback to greedy
        if hasattr(self.crf, 'decode'):
            return self.crf.decode(emissions, mask=mask)
        else:
            # fallback: greedy argmax
            return emissions.argmax(dim=-1).tolist()
