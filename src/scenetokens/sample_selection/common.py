"""Shared helpers for sample selection strategies."""

from typing import Any

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray


# Exponent used in Gumbel sorting for zero-weight samples — large enough to strongly deprioritize
# them without completely zeroing out their probability via uniform**inf.
_GUMBEL_LARGE_EXPONENT: float = 8.0


def aggregate_selected_samples(selected_samples: dict[Any, Any]) -> None:
    """Helper function to aggregate the sample IDs to keep and drop across groups (tokens or clusters) into a single
    list of samples to keep and drop. Mutates the input dictionary in place.

    Args:
        selected_samples: a dictionary containing the sample selection results per group.
    """
    keep = []
    drop = []
    for samples in selected_samples.values():
        keep += samples["keep"]
        drop += samples["drop"]
    selected_samples["keep"] = keep
    selected_samples["drop"] = drop
    selected_samples["num_to_keep"] = len(keep)
    selected_samples["num_to_drop"] = len(drop)


def make_group_result(keep: list[Any], drop: list[Any]) -> dict[str, Any]:
    """Constructs a per-group selection result dict."""
    return {"keep": keep, "num_to_keep": len(keep), "drop": drop, "num_to_drop": len(drop)}


def compute_proportional_number_to_drop(
    total_number_to_drop: int, percentage: float, min_percentage: float, total_valid_percentage: float
) -> int:
    """Computes the proportional number of samples to drop for a group.

    Args:
        total_number_to_drop: the total number of samples to drop across all groups.
        percentage: the percentage of samples in the group.
        min_percentage: the min percentage threshold for a group to be considered valid for dropping samples.
        total_valid_percentage: the total percentage of samples across all valid groups.

    Returns:
        0 if percentage does not exceed min_percentage, favoring underrepresented groups by flooring rather than
        ceiling the per-group drop count.
    """
    return int(percentage * total_number_to_drop / total_valid_percentage) if percentage > min_percentage else 0


def weighted_sorting(
    samples: NDArray[object], weights: NDArray[np.float64], *, sort_ascending: bool = True
) -> tuple[NDArray[object], NDArray[np.float64]]:
    """Sorts the samples of an array using based on their weight values.

    Args:
        samples: a numpy array containing samples.
        weights: weights values in [0.0, 1.0] corresponding to each sample.
        sort_ascending: if 'True' it sorts the samples in ascending order so the lowest weight values appear first.

    Returns:
        samples: the sorted samples.
        weights: the sorted weights.
    """
    if len(samples) != len(weights):
        error_message = f"Size of samples {len(samples)} and weights {len(weights)} must be the same."
        raise ValueError(error_message)

    # Sort the sample indices based on the key values. When sort_ascending=True, the lowest weight values appear first.
    sorted_indices = np.argsort(weights) if sort_ascending else np.argsort(weights)[::-1]

    return samples[sorted_indices], weights[sorted_indices]


def weighted_sorting_gumbel(
    samples: NDArray[object],
    weights: NDArray[np.float64],
    generator: Generator,
    *,
    sort_ascending: bool = True,
    large_exponent: float = np.inf,
) -> tuple[NDArray[object], NDArray[np.float64]]:
    """Sorts the samples of an array using the Gumbel Max weighted sampling trick:
        https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/. Weights are assumed to be in [0, 1].

    Args:
        samples: a numpy array containing samples.
        weights: weights values in [0.0, 1.0] corresponding to each sample.
        generator: a random generator instance.
        sort_ascending: if 'True' it sorts the samples in ascending order, based on the key values.
        large_exponent: exponent value to use for samples whose weights are zero.

    Returns:
        samples: the sorted samples.
        weights: the sorted weights.
    """
    if len(samples) != len(weights):
        error_message = f"Size of samples {len(samples)} and weights {len(weights)} must be the same."
        raise ValueError(error_message)

    # Generate random numbers in [0, 1]
    uniform = generator.random(len(samples))

    # Calculate the exponent term (1 / W_i), if the weight of a sample is low its exponent to will be high.
    exponent = np.where(weights > 0.0, 1.0 / weights, large_exponent)

    # Calculate the priority values (uniform ** (1 / W_i)). Elements in 'uniform' raised to a large power (inf) will
    # result in 0.0.
    priority = uniform**exponent

    # Sort the sample indices based on the key values. If 'sort_ascending=False' higher priority values will show first.
    sorted_indices = np.argsort(priority) if sort_ascending else np.argsort(priority)[::-1]

    return samples[sorted_indices], weights[sorted_indices]


def sort_ids_by_score(
    ids: NDArray[object],
    scores: NDArray[np.float64],
    sorting_strategy: str,
    seed: int,
) -> tuple[NDArray[object], NDArray[np.float64]]:
    """Sorts IDs by score so that the lowest-priority candidates (to drop) appear first.

    For 'gumbel': uses (1 - scores) with the Gumbel Max trick so high-scoring (typical) samples are softly
    deprioritized. For other strategies: sorts by raw scores descending so the highest-scoring IDs appear last.
    """
    if sorting_strategy == "gumbel":
        generator = default_rng(seed)
        return weighted_sorting_gumbel(
            ids, 1.0 - scores, generator, sort_ascending=True, large_exponent=_GUMBEL_LARGE_EXPONENT
        )
    return weighted_sorting(ids, scores, sort_ascending=False)
