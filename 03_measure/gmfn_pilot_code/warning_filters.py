"""Targeted warning filters shared by evaluation entry points."""

import os
import warnings


# Install before importing torch, including in spawned DataLoader workers.
warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\. Please install nvidia-ml-py instead\.",
    category=FutureWarning,
    module=r"torch\.cuda(?:\..*)?",
)

# PyCharm's debugger helpers can import pkg_resources in DataLoader workers.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\.",
    category=UserWarning,
)

# Spawned interpreters load debugger helpers before importing this module.
# Pass the same narrow filter to them at interpreter startup.
_worker_filter = "ignore:pkg_resources is deprecated as an API.:UserWarning"
_existing_filters = os.environ.get("PYTHONWARNINGS", "")
if _worker_filter not in _existing_filters.split(","):
    os.environ["PYTHONWARNINGS"] = ",".join(
        value for value in (_existing_filters, _worker_filter) if value
    )
