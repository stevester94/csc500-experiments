#! /bin/sh
export PYTHONPATH=/usr/local/lib/python3/dist-packages:/usr/local/lib/python3.6/dist-packages
cat << EOF | ./run.sh -
{
  "experiment_name": "Manual Experiment",
  "lr": 0.001,
  "n_epoch": 100,
  "batch_size": 128,
  "patience": 10,
  "device": "cuda",
  "source_domains": [
    4,
    6,
    8
  ],
  "target_domains": [
    2,
    10,
    12,
    14,
    16,
    18,
    20
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
    }
  ],
  "u_net": [
    {
      "class": "nnReshape",
      "kargs": {
        "shape": [
          -1,
          1
        ]
      }
    },
    {
      "class": "Linear",
      "kargs": {
        "in_features": 1,
        "out_features": 10
      }
    }
  ],
  "merge_net": [
    {
      "class": "Linear",
      "kargs": {
        "in_features": 2910,
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
    }
  ],
  "class_net": [
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
  "domain_net": [
    {
      "class": "Linear",
      "kargs": {
        "in_features": 256,
        "out_features": 1
      }
    },
    {
      "class": "nnClamp",
      "kargs": {
        "min": 0,
        "max": 1
      }
    },
    {
      "class": "Flatten",
      "kargs": {
        "start_dim": 0
      }
    }
  ],
  "num_examples": 160000,
  "seed": 1316,
  "alpha": 1
}
EOF