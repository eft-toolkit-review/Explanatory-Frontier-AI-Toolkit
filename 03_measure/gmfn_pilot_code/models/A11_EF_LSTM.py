import torch
import torch.nn as nn
import torch.nn.functional as F

class A11_EF_LSTM(nn.Module):
    """
    early fusion using lstm
    """
    def __init__(self, args):
        super(A11_EF_LSTM, self).__init__()
        text_in, audio_in, video_in = args.feature_dims

        in_size = text_in + audio_in + video_in
        hidden_size = args.hidden_dims
        num_layers = args.num_layers
        dropout = args.dropout
        output_dim =  1


        # BatchNorm will be created lazily on first forward pass with correct seq_len
        self.norm = None
        # Improvement: reduce internal LSTM dropout and increase external regularization
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers=num_layers,
                           dropout=min(dropout, 0.3) if num_layers > 1 else 0,
                           bidirectional=True, batch_first=True)

        # Improvement: use separate dropout layers with a higher dropout rate
        self.dropout_lstm = nn.Dropout(dropout)
        self.dropout_fc = nn.Dropout(min(dropout + 0.1, 0.7))

        # Improvement: correct the input dimension for the bidirectional LSTM (*2)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_dim)

        # Improvement: add LayerNorm for additional regularization
        self.layer_norm = nn.LayerNorm(hidden_size)



    def forward(self, text_x, audio_x, video_x):
        # Align all modalities to the same sequence length (use text length as reference)
        min_len = min(text_x.shape[1], audio_x.shape[1], video_x.shape[1])
        text_x = text_x[:, :min_len, :]
        audio_x = audio_x[:, :min_len, :]
        video_x = video_x[:, :min_len, :]

        x = torch.cat([text_x, audio_x, video_x], dim=-1)

        # Lazy init BatchNorm1d: expects (N, C, L), input is (N, L, C)
        if self.norm is None:
            self.norm = nn.BatchNorm1d(x.shape[2]).to(x.device)
        # Apply norm per feature: transpose -> norm -> transpose back
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        # Improvement: apply dropout to inputs for additional regularization
        x = self.dropout_lstm(x)

        _, final_states = self.lstm(x)

        # Improvement: concatenate forward and backward hidden states for the bidirectional LSTM
        # final_states[0] shape: (num_layers * 2, batch, hidden_size)
        # Get the forward and backward hidden states of the last layer
        forward_hidden = final_states[0][-2]  # Last-layer forward state
        backward_hidden = final_states[0][-1]  # Last-layer backward state
        x = torch.cat([forward_hidden, backward_hidden], dim=-1)

        x = self.dropout_lstm(x)
        x = F.relu(self.linear(x), inplace=True)

        # Improvement: add LayerNorm
        x = self.layer_norm(x)
        x = self.dropout_fc(x)

        output = self.out(x)
        res = {
            'M': output
        }
        return res
