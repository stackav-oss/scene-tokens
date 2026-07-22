# Model Training

## Training a single experiment

To run a training experiment:

```bash
uv run -m scenetokens.train model=[model_name]
```

where `model_name`: either of `wayformer`, `scenetransformer`, `scenetokens`, `safe_scenetokens`, `mtr` or `autobot`.
The model name needs to be specified.

Additional command line arguments:
* `logger`: either of `mlflow`, `neptune`, `tensorboard`, `wandb`, `csv` or `many_loggers`. The `many_loggers` composition is defined in `src/scenetokens/configs/logger/many_loggers.yaml`. Specific parameters might need to be set for some loggers. **Default** value is `many_loggers`.
* `scenario`: either of `waymo` or `nuscenes`. This will simply set the scenario sequence partition. **Default** value is `waymo`, which will partition the scenario into 1.1 seconds of history and 8 seconds for prediction.
* `paths`: either of `waymo`, `causal_agents`, `safeshift`, `safeshift_causal`, `ego_safeshift_causal`, or `open_scenario`. Each specifies the paths to the train/val/test data. **Default** value is `causal_agents`.
* `trainer`: either of `cpu`, `ddp`, `gpu` or `mps`. **Default** value is `gpu`.
* `dataset`: This specifies the input data representation. Currently supported values are `waymo` and `open_scenario`. See this [doc](./DATA_PREPARATION.md) for more details on how to prepare Waymo data.

## OpenScenario training

Use `dataset=open_scenario` and `paths=open_scenario` for OpenScenario datasets materialised under `${paths.base_path}/processed/${paths.tag}`.
The OpenScenario loader preserves physical time by sampling at `dataset.config.target_frequency_hz`, expressed in hertz.
It derives the closest fixed integer stride from the source timestamps, then truncates or rejects scenarios according to `scenario.past_len + scenario.future_len`.
It does not rescale each scenario to a variable timestep interval.

```bash
uv run -m scenetokens.train \
  model=scenetokens \
  dataset=open_scenario \
  paths=open_scenario \
  paths.tag=[dataset_tag] \
  dataset.config.target_frequency_hz=[frequency_hz] \
  test=false
```

The configured target frequency should match the intended physical cadence of the materialised dataset.
The default Waymo horizon remains `past_len=11` and `future_len=80`, requiring `91` sampled frames per cached scenario.
Set `test=false` when the materialised dataset only contains training and validation splits.
Set model-specific options through explicit `model.config` overrides, for example `model.config.token_conditioning=true`.

## Logging Details

#### CSV (Default through `many_loggers`)
Outputs will be saved to `out/logs/runs/date/experiment_name/csv`.

#### MLflow
MLflow is available through `logger=mlflow` and needs **tracking_uri** specification, as:

```bash
uv run -m scenetokens.train model=wayformer logger.mlflow.tracking_uri=[uri]
```

#### Tensorboard
To visualize logs:
```bash
uv run tensorboard --logdir out/ --host [host-address] --port [port]
```

#### Other
Other logger configurations, including `neptune` and `wandb`, are backed by PyTorch Lightning support. See this [link](https://lightning.ai/docs/pytorch/stable/api_references.html#loggers) for reference.

## Evaluating a single experiment

To run an evaluation, specify any additional config arguments as above and a checkpoint name.
```bash
uv run -m scenetokens.eval ckpt_path=/path/to/the/ckpt.pth model=[model_name]
```

## Debugging

There are various debugging configurations which can be enabled by adding `debug=[debug_name]` to the command, where `debug_name` is either of:
* `default`: runs one epoch on debug mode on cpu.
* `fdr`: runs 1 train, 1 validation and 1 testing step.
* `limit`: runs n epochs with 1% of the training data and 5% of the val/test data.
* `overfit`: runs n epochs to overfit on b batches.
* `profiler`: runs a performance profiler experiment.

Example, running a fast development run for an OpenScenario dataset:
```bash
uv run -m scenetokens.train model=scenetokens dataset=open_scenario paths=open_scenario paths.tag=[dataset_tag] debug=fdr test=false
```

Example, running the profiler:
```bash
uv run -m scenetokens.train model=wayformer debug=profiler
```

# Multirun training (Parameter Sweeps)

To run a sweep of experiments use `-m` and specify in the command line the parameter(s) to be swept. For example:
```bash
uv run -m scenetokens.train -m model=[model_name] model.config.num_classes=10,20,50,100
```
This will launch 4 sequential experiments where the value `num_classes` will be set to 10, 20, 50 and 100, respectively. The experiment logs will be saved to `out/logs/multiruns` instead of `out/logs/runs/`.
