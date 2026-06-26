# Model Training

## Single Experiment

Run a training experiment with Hydra configuration overrides.

```bash
uv run -m scenetokens.train model=[model_name]
```

The `model` value selects the model configuration.
Supported values include `wayformer`, `scenetransformer`, `scenetokens`, `scenetokens_student`, `safe_scenetokens`, `mtr`, and `autobot`, subject to the configuration files present in `src/scenetokens/configs/model`.
The `scenetokens_student` option is a Student-equivalent alias over the current renamed `SceneTokens` implementation.

## Common Overrides

- `logger`: Selects the logger configuration, such as `csv`, `mlflow`, `neptune`, `tensorboard`, `wandb`, or `many_loggers`.
- `scenario`: Selects the model horizon configuration.
- `paths`: Selects the input and output path family.
- `trainer`: Selects the PyTorch Lightning trainer configuration, such as `cpu`, `ddp`, `gpu`, or `mps`.
- `dataset`: Selects the dataset representation, such as `waymo` or `open_scenario`.

## OpenScenario Training

Use `dataset=open_scenario` and `paths=open_scenario` for OpenScenario datasets materialised under `${paths.base_path}/processed/${paths.tag}`.
The OpenScenario loader preserves physical time by sampling at `dataset.config.target_frequency_hz`, expressed in hertz.
It derives the closest fixed integer stride from the source timestamps, then truncates or rejects scenarios according to `scenario.past_len + scenario.future_len`.
It does not rescale each scenario to a variable timestep interval.

```bash
uv run -m scenetokens.train \
  model=scenetokens_student \
  dataset=open_scenario \
  paths=open_scenario \
  paths.tag=[dataset_tag] \
  dataset.config.target_frequency_hz=[frequency_hz] \
  test=false
```

The configured target frequency should match the intended physical cadence of the materialised dataset.
The default Waymo horizon remains `past_len=11` and `future_len=80`, requiring `91` sampled frames per cached scenario.

## Logging

CSV logs are written under the configured Hydra output directory.
MLflow requires an explicit tracking URI when the `mlflow` logger is used.

```bash
uv run -m scenetokens.train model=wayformer logger.mlflow.tracking_uri=[uri]
```

TensorBoard can visualise logs from the configured output directory.

```bash
uv run tensorboard --logdir out/ --host [host-address] --port [port]
```

## Evaluation

Run evaluation by providing a checkpoint path and the same model and data overrides used for training.

```bash
uv run -m scenetokens.eval ckpt_path=/path/to/the/ckpt.pth model=[model_name]
```

## Debugging

Debug configurations can be enabled with `debug=[debug_name]`.
The `fdr` configuration runs one train step and one validation step.
The `limit` configuration runs a small bounded fraction of the train and validation sets.

```bash
uv run -m scenetokens.train model=scenetokens_student dataset=open_scenario paths=open_scenario paths.tag=[dataset_tag] debug=fdr test=false
```

## Multirun Training

Use `-m` for Hydra multirun sweeps.

```bash
uv run -m scenetokens.train -m model=[model_name] model.config.num_classes=10,20,50,100
```
