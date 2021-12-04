#! /bin/sh
export PYTHONPATH=/usr/local/lib/python3/dist-packages:/usr/local/lib/python3.6/dist-packages
cat << EOF | ./run.sh -
{
  "experiment_name": "Prototypical ORACLE Base Case",
  "lr": 0.001,
  "desired_serial_numbers": [
    "3123D52",
    "3123D65",
    "3123D79",
    "3123D80"
  ],
  "source_domains": [
    2
  ],
  "window_stride": 50,
  "window_length": 256,
  "desired_runs": [
    1
  ],
  "num_examples_per_device": 75000,
  "n_val_tasks": 1000,
  "n_test_tasks": 100,
  "validation_frequency": 1000,
  "n_epoch": 100,
  "patience": 10,
  "seed": 1337,
  "n_query": 50,
  "n_train_tasks": 5000,
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
      "class": "Flatten",
      "kargs": {}
    },
    {
      "class": "Linear",
      "kargs": {
        "in_features": 5800,
        "out_features": 512
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
        "in_features": 512,
        "out_features": 512
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
        "in_features": 512,
        "out_features": 512
      }
    }
  ]
}
EOF