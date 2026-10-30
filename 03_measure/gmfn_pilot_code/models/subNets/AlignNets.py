import torch
import torch.nn as nn

__all__ = ['AlignSubNet']


class AlignSubNet(nn.Module):
    """Align modality sequences to the text length using average pooling."""

    def __init__(self, args):
        super().__init__()
        self.dst_len = args.seq_lens[0]

    def get_seq_len(self):
        return self.dst_len

    def _align(self, x):
        seq_len = x.size(1)
        if seq_len == self.dst_len:
            return x

        pad_len = (-seq_len) % self.dst_len
        pool_size = (seq_len + pad_len) // self.dst_len
        padding = x[:, -1:, :].expand(-1, pad_len, -1)
        x = torch.cat((x, padding), dim=1)
        return x.reshape(x.size(0), pool_size, self.dst_len, -1).mean(dim=1)

    def forward(self, text_x, audio_x, video_x):
        if text_x.size(1) == audio_x.size(1) == video_x.size(1):
            return text_x, audio_x, video_x
        return self._align(text_x), self._align(audio_x), self._align(video_x)
