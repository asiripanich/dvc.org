# Hugging Face

DVCLive allows you to add experiment tracking capabilities to your
[Hugging Face](https://huggingface.co/) projects.

## Usage

<p align='center'>
  <a href="https://colab.research.google.com/github/iterative/dvclive/blob/main/examples/DVCLive-HuggingFace.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" />
  </a>
</p>

Include the
[`DVCLiveCallback`](https://github.com/iterative/dvclive/blob/main/src/dvclive/huggingface.py)
in the callbacks list passed to your
[`Trainer`](https://huggingface.co/transformers/main_classes/trainer.html):

```python
from dvclive.huggingface import DVCLiveCallback

...

 trainer = Trainer(
    model, args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.add_callback(DVCLiveCallback(save_dvc_exp=True))
trainer.train()
```

## Parameters

- `live` - (`None` by default) - Optional [`Live`] instance. If `None`, a new
  instance will be created using `**kwargs`.

- `log_model` - (`None` by default) - use
  [`live.log_artifact()`](/doc/dvclive/live/log_artifact) to log checkpoints
  created by the
  [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer#checkpoints).

  - if `log_model == 'all'`, all checkpoints are logged during training.
    [`live.log_artifact()`] is called with `Trainer.output_dir`.

  - if `log_model == 'last'`, the final checkpoint is logged at the end of
    training. A copy of the final checkpoint will be saved in
    `Live.artifacs_dir`. [`live.log_artifact()`] is called with `type="model"`
    and `copy=True`.

    If you set `Trainer.load_best_model_at_end` to `True`, the checkpoint logged
    will correspond to the best one.

  - if `log_model is None` (default), no checkpoint is logged.

- `**kwargs` - Any additional arguments will be used to instantiate a new
  [`Live`] instance. If `live` is used, the arguments are ignored.

## Examples

- Using `live` to pass an existing [`Live`] instance.

```python
from dvclive import Live
from dvclive.huggingface import DVCLiveCallback

with Live("custom_dir", save_dvc_exp=True) as live:
    trainer = Trainer(
        model, args,
        train_dataset=train_data, eval_dataset=eval_data, tokenizer=tokenizer)
    trainer.add_callback(
        DVCLiveCallback(live=live))

    # Log additional metrics after training
    live.log_metric("summary_metric", 1.0, plot=False)
```

- Using `**kwargs` to customize the new [`Live`] instance.

```python
trainer.add_callback(
    DVCLiveCallback(save_dvc_exp=True, dir="custom_dir"))
```

## Output format

Each metric will be logged to:

```py
{Live.plots_dir}/metrics/{split}/{metric}.tsv
```

Where:

- `{Live.plots_dir}` is defined in [`Live`].
- `{split}` can be either `train` or `eval`.
- `{metric}` is the name provided by the framework.

[`live`]: /doc/dvclive/live
