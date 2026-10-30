"""A1.2 Alignment and standardization (illustrative). """
import numpy as np


def align_to_tokens(token_start, token_end, frame_times, frames):
    """Mean of the frames of one modality that fall inside each token's interval (E1).

    token_start, token_end: (n_tokens,) seconds.   frame_times: (n_frames,) seconds.
    frames: (n_frames, d).   Returns (n_tokens, d).
    Rule for an interval with no frame (illustrative; to be fixed in the registry): take the
    frame nearest in time to the interval's midpoint.
    """
    token_start, token_end = np.asarray(token_start, float), np.asarray(token_end, float)
    frame_times, frames = np.asarray(frame_times, float), np.asarray(frames, float)
    out = np.empty((len(token_start), frames.shape[1]))
    for i, (a, b) in enumerate(zip(token_start, token_end)):
        inside = (frame_times >= a) & (frame_times < b)
        if inside.any():
            out[i] = frames[inside].mean(axis=0)
        else:
            out[i] = frames[np.argmin(np.abs(frame_times - (a + b) / 2))]
    return out


def fit_standardizer(train_sequences):
    """Mean and standard deviation of each feature over all tokens of the TRAINING clips only
    (the training set in a single split, the training folds in cross-fitting)."""
    x = np.concatenate([np.asarray(s, float) for s in train_sequences], axis=0)
    sd = x.std(axis=0)
    return {"mean": x.mean(axis=0), "std": np.where(sd > 0, sd, 1.0)}


def standardize(sequence, stats):
    return (np.asarray(sequence, float) - stats["mean"]) / stats["std"]
