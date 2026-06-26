"""OpenScenario dataset loader for pre-processed Scenario objects."""

import pickle
from pathlib import Path
from typing import Final, final, override

import numpy as np
from characterization.schemas import DynamicMapData, Scenario
from omegaconf import DictConfig

from scenetokens.datasets.base_dataset import BaseDataset
from scenetokens.utils import pylogger


_LOGGER: Final = pylogger.get_pylogger(__name__)
_FREQUENCY_TOLERANCE_HZ: Final = 1e-3
_MIN_SAMPLE_INTERVAL: Final = 1
_MIN_TIMESTAMPS_FOR_FREQUENCY: Final = 2
_MIN_TARGET_FREQUENCY_HZ: Final = 0.0


@final
class _OpenScenarioWindowError(ValueError):
    """Raised when an OpenScenario file cannot provide the configured model horizon."""


@final
class OpenScenarioDataset(BaseDataset):
    """Dataset loader for OpenScenario format.

    This dataset expects scenarios to already be saved as Scenario objects in pickle files.
    The preprocessing step should have already converted source data into the Scenario
    schema format and saved them as .pkl files.

    Directory structure expected:
        /datasets/open_scenario/processed/<variant>/
        ├── training/
        │   ├── scenario_001.pkl  # Each contains a complete Scenario object
        │   ├── scenario_002.pkl
        │   └── ...
        ├── validation/
        │   └── *.pkl
        └── testing/
            └── *.pkl

    OpenScenario logs may contain different physical durations.
    The loader preserves the source time scale by deriving a fixed integer stride from `target_frequency_hz`, then
    enforces the configured model horizon by truncating long scenarios and skipping scenarios that are too short.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the OpenScenario dataset.

        Args:
            config: Dataset configuration from Hydra config.
        """
        super().__init__(config)

    @override
    def load_and_process_scenario(self, path: Path) -> list[dict[str, object]] | None:
        """Process a scenario and skip expected OpenScenario horizon failures."""
        try:
            return super().load_and_process_scenario(path)
        except _OpenScenarioWindowError as exc:
            _LOGGER.warning("Skipping OpenScenario file %s: %s", path, exc)
            return None

    @override
    def load_as_open_scenario(self, path: Path) -> Scenario:
        """Loads a pre-processed Scenario object from a pickle file.

        Args:
            path: Full path to the scenario pickle file.

        Returns:
            The normalised Scenario object.
        """
        with path.open("rb") as f:
            scenario: Scenario = pickle.load(f)
        return self._normalise_temporal_window(scenario)

    def _normalise_temporal_window(self, scenario: Scenario) -> Scenario:
        """Preserve a fixed physical sampling frequency and enforce the configured model horizon."""
        num_timestamp_steps = len(scenario.metadata.timestamps_seconds)
        num_trajectory_steps = scenario.agent_data.agent_trajectories.shape[1]
        if num_timestamp_steps != num_trajectory_steps:
            error_message = (
                f"Scenario {scenario.metadata.scenario_id} has {num_timestamp_steps} timestamps but "
                f"{num_trajectory_steps} trajectory steps."
            )
            raise ValueError(error_message)

        source_frequency_hz = self._estimate_frequency_hz(
            [float(timestamp_seconds) for timestamp_seconds in scenario.metadata.timestamps_seconds]
        )
        sample_interval = self._sample_interval_for_target_frequency_hz(scenario, source_frequency_hz)
        sampled_indices = np.arange(0, num_trajectory_steps, sample_interval, dtype=np.int64)
        if sampled_indices.size < self.total_steps:
            error_message = (
                f"{sampled_indices.size} sampled steps available at target_frequency_hz="
                f"{self._target_frequency_hz(source_frequency_hz):.6g}; {self.total_steps} required for "
                f"past_len={self.past_len}, future_len={self.future_len}."
            )
            raise _OpenScenarioWindowError(error_message)

        window_indices = sampled_indices[: self.total_steps]
        timestamps_seconds = [float(scenario.metadata.timestamps_seconds[int(index)]) for index in window_indices]
        agent_data = scenario.agent_data.model_copy(
            update={"agent_trajectories": scenario.agent_data.agent_trajectories[:, window_indices, :]}
        )
        metadata = scenario.metadata.model_copy(
            update={
                "timestamps_seconds": timestamps_seconds,
                "frequency_hz": self._estimate_frequency_hz(timestamps_seconds),
                "current_time_index": self.current_time_idx,
                "track_length": self.total_steps,
                "dataset": self._dataset_name(scenario),
            }
        )
        return scenario.model_copy(
            update={
                "metadata": metadata,
                "agent_data": agent_data,
                "dynamic_map_data": self._slice_dynamic_map_data(scenario.dynamic_map_data, window_indices),
            }
        )

    def _sample_interval_for_target_frequency_hz(self, scenario: Scenario, source_frequency_hz: float) -> int:
        """Return the fixed integer stride that best approximates the requested physical frequency."""
        if source_frequency_hz <= _MIN_TARGET_FREQUENCY_HZ:
            error_message = f"Scenario {scenario.metadata.scenario_id} has non-positive source frequency."
            raise ValueError(error_message)

        target_frequency_hz = self._target_frequency_hz(source_frequency_hz)
        if target_frequency_hz <= _MIN_TARGET_FREQUENCY_HZ:
            error_message = f"target_frequency_hz must be positive, got {target_frequency_hz}."
            raise ValueError(error_message)
        if target_frequency_hz - source_frequency_hz > _FREQUENCY_TOLERANCE_HZ:
            error_message = (
                f"target_frequency_hz={target_frequency_hz:.6g} requires upsampling source frequency "
                f"{source_frequency_hz:.6g} for scenario {scenario.metadata.scenario_id}."
            )
            raise ValueError(error_message)

        sample_interval = max(_MIN_SAMPLE_INTERVAL, round(source_frequency_hz / target_frequency_hz))
        achieved_frequency_hz = source_frequency_hz / sample_interval
        if abs(achieved_frequency_hz - target_frequency_hz) > _FREQUENCY_TOLERANCE_HZ:
            _LOGGER.warning(
                "Scenario %s requested target_frequency_hz=%.6g, but source frequency %.6g supports %.6g with "
                "integer stride %s.",
                scenario.metadata.scenario_id,
                target_frequency_hz,
                source_frequency_hz,
                achieved_frequency_hz,
                sample_interval,
            )
        return sample_interval

    def _target_frequency_hz(self, source_frequency_hz: float) -> float:
        """Return the configured target frequency, defaulting to the source frequency."""
        target_frequency_hz = self.config.get("target_frequency_hz", None)
        if target_frequency_hz is None:
            return source_frequency_hz
        return float(target_frequency_hz)

    def _dataset_name(self, scenario: Scenario) -> str:
        """Return the dataset name propagated into cached model samples."""
        if self.subset_data_tag is None:
            return scenario.metadata.dataset
        return f"open_scenario-{self.subset_data_tag}"

    @staticmethod
    def _estimate_frequency_hz(timestamps_seconds: list[float]) -> float:
        """Estimate frequency from fixed-scale sampled timestamps."""
        if len(timestamps_seconds) < _MIN_TIMESTAMPS_FOR_FREQUENCY:
            return 0.0
        timestep_seconds = np.diff(np.asarray(timestamps_seconds, dtype=np.float64))
        mean_timestep_seconds = float(np.mean(timestep_seconds))
        if mean_timestep_seconds <= 0.0:
            return 0.0
        return 1.0 / mean_timestep_seconds

    @staticmethod
    def _slice_dynamic_map_data(dynamic_map_data: DynamicMapData | None, indices: np.ndarray) -> DynamicMapData | None:
        """Slice temporal dynamic-map fields when they are available."""
        if dynamic_map_data is None:
            return None
        return dynamic_map_data.model_copy(
            update={
                "stop_points": OpenScenarioDataset._slice_temporal_values(dynamic_map_data.stop_points, indices),
                "lane_ids": OpenScenarioDataset._slice_temporal_values(dynamic_map_data.lane_ids, indices),
                "states": OpenScenarioDataset._slice_temporal_values(dynamic_map_data.states, indices),
            }
        )

    @staticmethod
    def _slice_temporal_values(values: list[object] | None, indices: np.ndarray) -> list[object] | None:
        """Slice an optional temporal list with integer indices."""
        if values is None:
            return None
        return [values[int(index)] for index in indices]
