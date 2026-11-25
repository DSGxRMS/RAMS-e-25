"""Sampling seed shared by the controller and its run recorder."""
import os


def sampling_seed():
    raw = os.environ.get("MPPI_SEED", "42")
    try:
        seed = int(raw)
    except ValueError:
        raise ValueError("MPPI_SEED must be an integer from 0 to 2^63 - 1") from None
    if not 0 <= seed < 2**63:
        raise ValueError("MPPI_SEED must be an integer from 0 to 2^63 - 1")
    return seed
