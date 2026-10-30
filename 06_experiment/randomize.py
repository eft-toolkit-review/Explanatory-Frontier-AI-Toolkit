"""A5.1 Balanced random assignment (illustrative).

Default design: M = 48 source clips, J = 8 clips per participant (four high, four low), 360 participants.
Every clip is rated 60 times, 30 per version; no participant sees the same clip twice; clip order is random.
The registry seed (master seed combined with the script name) is passed in by the caller.
"""
import numpy as np


def assign(participants=360, clips=48, J=8, seed=0, max_tries=100000):
    half = J // 2
    if clips % half or participants % (clips // half):
        raise ValueError("this sketch covers designs where clips/(J/2) divides the number of participants")
    groups = clips // half                                      # 12 groups of 4 clips
    rng = np.random.default_rng(seed)
    rows = []
    for block in range(participants // groups):                 # each block uses every clip once as high, once as low
        for _ in range(max_tries):
            hi = rng.permutation(clips).reshape(groups, half)   # random partition into groups, one per participant
            lo = rng.permutation(clips).reshape(groups, half)
            match = rng.permutation(groups)
            if all(not set(hi[g]) & set(lo[match[g]]) for g in range(groups)):
                break
        else:
            raise RuntimeError("no valid pairing found")
        for g in range(groups):
            trials = [(int(c), 1) for c in hi[g]] + [(int(c), 0) for c in lo[match[g]]]   # (clip, high = 1)
            rows.append([trials[i] for i in rng.permutation(J)])                           # random clip order
    order = rng.permutation(len(rows))                          # participant ids are not tied to blocks
    return [rows[i] for i in order]
