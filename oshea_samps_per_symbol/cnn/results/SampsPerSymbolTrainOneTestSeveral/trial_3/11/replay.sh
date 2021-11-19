#! /bin/sh
export PYTHONPATH=/usr/local/lib/python3/dist-packages:/usr/local/lib/python3.6/dist-packages
cat << EOF | ./run.sh -
{
  "experiment_name": "CNN SampsPerSymbol Train One Test Several Trial 3",
  "lr": 0.001,
  "n_epoch": 1000,
  "batch_size": 256,
  "patience": 10,
  "device": "cuda",
  "source_domains": [
    8
  ],
  "target_domains": [
    2,
    6,
    10,
    12
  ],
  "x_net": [
    {
      "class": "Conv1d",
      "kargs": {
        "in_channels": 2,
        "out_channels": 50,
        "kernel_size": 7,
        "stride": 1,
        "padding": 0
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "Conv1d",
      "kargs": {
        "in_channels": 50,
        "out_channels": 50,
        "kernel_size": 7,
        "stride": 2,
        "padding": 0
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "Dropout",
      "kargs": {
        "p": 0.5
      }
    },
    {
      "class": "Flatten",
      "kargs": {}
    },
    {
      "class": "Linear",
      "kargs": {
        "in_features": 2900,
        "out_features": 256
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "Dropout",
      "kargs": {
        "p": 0.5
      }
    },
    {
      "class": "Linear",
      "kargs": {
        "in_features": 256,
        "out_features": 80
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "Linear",
      "kargs": {
        "in_features": 80,
        "out_features": 9
      }
    }
  ],
  "seed": 3854
}
EOF