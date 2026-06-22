# CIFAR-10-C Stream Dataset Config

`CIFAR10CStreamDataset` builds an ordered stream from local CIFAR-10-C `.npy` files.
Use it through `DatasetFactory` with `dataset.name: cifar10c_stream`, or directly:

```python
from data import CIFAR10CStreamDataset

dataset = CIFAR10CStreamDataset.from_yaml("configs/cifar10c_stream.yaml")
```

## Minimal YAML

```yaml
dataset:
  name: cifar10c_stream
  data_dir: data/CIFAR-10-C
  clean_root: data
  segments:
    - corruption: clean
      count: 200
    - corruption: gaussian_noise
      count: 200
      severity: 1
    - corruption: shot_noise
      count: 200
      severity: 2
```

This creates a stream with 200 clean CIFAR-10 test samples, followed by 200
`gaussian_noise` samples, followed by 200 `shot_noise` samples.

## Dataset Settings

`name`: must be `cifar10c_stream`.

`data_dir`: directory containing CIFAR-10-C files such as `gaussian_noise.npy`,
`shot_noise.npy`, and `labels.npy`. The repository download script writes to
`data/CIFAR-10-C`.

`labels_file`: label file inside `data_dir`. Defaults to `labels.npy`.

`severity`: default CIFAR-10-C severity for segments that do not override it.
Valid values are `1`, `2`, `3`, `4`, and `5`.

`segments`: ordered list of stream blocks. Each block is appended to the stream
exactly in YAML order.

`channel_first`: when `true`, images are returned as `(C, H, W)`. When `false`,
images are returned as `(H, W, C)`. Defaults to `false`.

`normalize`: when `true`, image values are converted to floats in `[0, 1]`.
When `false`, values are only cast to `dtype`. Defaults to `false`.

`dtype`: NumPy dtype used for returned image arrays. Defaults to `float32`.

## Clean CIFAR-10 Settings

Clean segments use `corruption: clean` or `corruption: clear`.

`clean_root`: root directory passed to `torchvision.datasets.CIFAR10` for the
clean CIFAR-10 test split. Defaults to the parent of `data_dir`. The loader
first checks for a local CIFAR-10 test split there; if it is missing,
torchvision downloads it automatically.

`clean_data_file` and `clean_labels_file`: optional local `.npy` files for clean
images and labels. If these are set, torchvision is not used. This is useful in
offline or test environments.

## Segment Settings

Each item in `segments` supports:

`corruption`: CIFAR-10-C corruption name without `.npy`, for example
`gaussian_noise`, `shot_noise`, `motion_blur`, or `snow`. Use `clean`/`clear`
for the clean CIFAR-10 test split.

`name`: optional display name stored in sample metadata. If omitted, the
corruption name is used.

`count`: number of samples to take from this block. Required.

`severity`: CIFAR-10-C severity for this block. Overrides the dataset-level
`severity`. Ignored for clean segments.

`start`: starting index inside the selected clean/severity slice. Defaults to
`0`.

## Stream Items

Iteration yields dictionaries:

```python
{
    "x": image,
    "y": label,
    "metadata": {
        "index": 0,
        "segment": 0,
        "segment_name": "clean",
        "corruption": "clean",
        "severity": 1,
        "source_index": 0,
        "is_drift_point": False,
    },
}
```

`is_drift_point` is `true` on the first sample of every segment after the first.
`drift_positions` contains those stream indices.

See `docs/model_config.md` for the matching `SimpleCNN` model configuration
and CIFAR-10 training defaults used by the playground notebooks.
