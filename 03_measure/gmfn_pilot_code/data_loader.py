import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MMDataset(Dataset):
    """Test features with normalization statistics stored in the same PKL file."""

    def __init__(self, args):
        with open(args['featurePath'], 'rb') as f:
            data = pickle.load(f)

        self.text = data['text']
        self.audio = data['audio']
        self.vision = data['vision']
        self.info = data['info'].tolist()
        self.labels = {'M': (data['labels'] - 4).astype(np.float32)}

        for features in (self.text, self.audio, self.vision):
            features[np.isnan(features)] = 0

        stats = data['norm_stats']
        self.audio = self._normalize(self.audio, stats['audio_mean'], stats['audio_std'])
        self.vision = self._normalize(self.vision, stats['vision_mean'], stats['vision_std'])

        args['feature_dims'] = list(self.get_feature_dim())
        args['seq_lens'] = self.get_seq_len()

    @staticmethod
    def _normalize(features, mean, std):
        features = np.clip((features - mean) / std, -10.0, 10.0)
        features[np.isnan(features)] = 0
        return features

    def __len__(self):
        return self.text.shape[0]

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        return {
            'text': torch.Tensor(self.text[index]),
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'labels': {'M': torch.Tensor(self.labels['M'][index].reshape(-1))},
            'meta_info': self.info[index],
        }


def MMDataLoader(args, num_workers=0):
    dataset = MMDataset(args)
    return {
        'test': DataLoader(
            dataset,
            batch_size=args['batch_size'],
            num_workers=num_workers,
            shuffle=False,
            drop_last=True,
        )
    }
