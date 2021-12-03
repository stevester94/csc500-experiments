#! /usr/bin/env python3

import numpy as np
from math import floor

from torch.utils.data import DataLoader
from typing import Tuple
from torch import nn, optim
from torch.optim import Adam
from torchvision.models import resnet18
import torch


def my_compute_backbone_output_shape(backbone: nn.Module) -> Tuple[int]:
    """ 
    Compute the dimension of the feature space defined by a feature extractor.
    Args:
        backbone: feature extractor

    Returns:
        shape of the feature vector computed by the feature extractor for an instance

    """
    input_images = torch.ones((4, 2, 128))
    output = backbone(input_images)

    return tuple(output.shape[1:])
import easyfsl.utils; easyfsl.utils.compute_backbone_output_shape = my_compute_backbone_output_shape
from easyfsl.data_tools import EasySet, TaskSampler
from easyfsl.methods import PrototypicalNetworks, AbstractMetaLearner
from easyfsl.utils import sliding_average, compute_backbone_output_shape

from steves_utils.torch_sequential_builder import build_sequential
from steves_utils.dummy_cida_dataset import Dummy_CIDA_Dataset
from steves_utils.lazy_map import Lazy_Map



seed = 420

import random 
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

torch.use_deterministic_algorithms(True) 

from tqdm import tqdm


class Steves_Prototypical_Network(PrototypicalNetworks):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__(backbone)


    def fit(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        val_loader: DataLoader = None,
        validation_frequency: int = 1000,
    ):
        """
        Train the model on few-shot classification tasks.
        Args:
            train_loader: loads training data in the shape of few-shot classification tasks
            optimizer: optimizer to train the model
            val_loader: loads data from the validation set in the shape of few-shot classification
                tasks
            validation_frequency: number of training episodes between two validations
        """
        log_update_frequency = 10

        all_loss = []
        self.train()
        with tqdm(
            enumerate(train_loader), total=len(train_loader), desc="Meta-Training"
        ) as tqdm_train:
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_train:
                loss_value = self.fit_on_task(
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    optimizer,
                )
                all_loss.append(loss_value)

                # Log training loss in real time
                if episode_index % log_update_frequency == 0:
                    tqdm_train.set_postfix(
                        loss=sliding_average(all_loss, log_update_frequency)
                    )

                # Validation
                if val_loader:
                    if (episode_index + 1) % validation_frequency == 0:
                        print("")
                        self.validate(val_loader)

        return sum(all_loss) / len(all_loss)

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model on the validation set.
        Args:
            val_loader: loads data from the validation set in the shape of few-shot classification
                tasks
        Returns:
            average classification accuracy on the validation set
        """
        validation_accuracy = self.evaluate(val_loader)
        print("")
        print(f"Validation accuracy: {(100 * validation_accuracy):.2f}%")
        # If this was the best validation performance, we save the model state
        if validation_accuracy > self.best_validation_accuracy:
            print("Best validation accuracy so far!")
            self.best_model_state = self.state_dict()

        return validation_accuracy

    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluate the model on few-shot classification tasks
        Args:
            data_loader: loads data in the shape of few-shot classification tasks
        Returns:
            average classification accuracy
        """
        # We'll count everything and compute the ratio at the end
        total_predictions = 0
        correct_predictions = 0

        # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
        # no_grad() tells torch not to keep in memory the whole computational graph
        self.eval()
        with torch.no_grad():
            with tqdm(
                enumerate(data_loader), total=len(data_loader), desc="Evaluation"
            ) as tqdm_eval:
                for _, (
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    _,
                ) in tqdm_eval:
                    correct, total = self.evaluate_on_one_task(
                        support_images, support_labels, query_images, query_labels
                    )

                    total_predictions += total
                    correct_predictions += correct

                    # Log accuracy in real time
                    tqdm_eval.set_postfix(
                        accuracy=correct_predictions / total_predictions
                    )

        return correct_predictions / total_predictions

def build_Dummy_episodic_iterable(
    num_classes,
    num_examples_per_class,
    n_shot,
    n_query,
    n_train_tasks,
    n_val_tasks,
    n_test_tasks,
    seed,
):
    import copy 

    ds = Dummy_CIDA_Dataset(x_shape=[2,128], domains=[0], num_classes=num_classes, num_unique_examples_per_class=num_examples_per_class)
    ds = Lazy_Map(ds, lam=lambda ex: (torch.from_numpy(ex[0]), ex[1]))

    train_len = floor(len(ds)*0.7)
    val_len   = floor(len(ds)*0.15)
    test_len  = len(ds) - train_len - val_len

    train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed))

    train_ds.labels = [ex[1] for ex in train_ds]
    val_ds.labels   = [ex[1] for ex in val_ds]
    test_ds.labels  = [ex[1] for ex in test_ds]

    def wrap_in_dataloader(ds,n_tasks):
        sampler = TaskSampler(
                ds,
                n_way=num_classes,
                n_shot=n_shot,
                n_query=n_query,
                n_tasks=n_tasks
            )

        return torch.utils.data.DataLoader(
            ds,
            num_workers=6,
            persistent_workers=True,
            prefetch_factor=50,
            # pin_memory=True,
            batch_sampler=sampler,
            collate_fn=sampler.episodic_collate_fn
        )

    val_list = []
    for k in wrap_in_dataloader(val_ds, n_val_tasks):
        val_list.append(copy.deepcopy(k))

    test_list = []
    for k in wrap_in_dataloader(test_ds, n_test_tasks):
        test_list.append(copy.deepcopy(k))

    
    return (
        wrap_in_dataloader(train_ds, n_train_tasks),
        val_list,
        test_list,
    )


train_dl, val_dl, test_dl = build_Dummy_episodic_iterable(
    num_classes=4,
    num_examples_per_class=75000,
    n_shot=5,
    n_query=10,
    n_train_tasks=2500,
    n_val_tasks=500,
    n_test_tasks=500,
    seed=seed,
)

x_net = [
        {"class": "Flatten", "kargs": {}},

        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 1024}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},

        {"class": "Linear", "kargs": {"in_features": 1024, "out_features": 1024}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},

        {"class": "Linear", "kargs": {"in_features": 1024, "out_features": 512}},
]

x_net = [
        # {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0, "groups":2 },},
        {"class": "Conv1d", "kargs": { "in_channels":2, "out_channels":50, "kernel_size":7, "stride":1, "padding":0,  },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Conv1d", "kargs": { "in_channels":50, "out_channels":50, "kernel_size":7, "stride":1, "padding":0 },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},
        {"class": "Flatten", "kargs": {}},

        {"class": "Linear", "kargs": {"in_features": 5800, "out_features": 1024}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},

        {"class": "Linear", "kargs": {"in_features": 1024, "out_features": 1024}},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "Dropout", "kargs": {"p": 0.5}},

        {"class": "Linear", "kargs": {"in_features": 1024, "out_features": 512}},
]


x_net = build_sequential(x_net)

model = Steves_Prototypical_Network(x_net).cuda()
optimizer = Adam(params=model.parameters())






NUM_EPOCHS = 25

epoch_train_loss = model.fit(train_dl, optimizer, val_loader=val_dl, validation_frequency=500)


# for epoch in range(NUM_EPOCHS):
#     epoch_train_loss = model.fit(train_dl, optimizer, val_loader=val_dl, validation_frequency=500)
#     accuracy = model.evaluate(test_dl)

#     print(epoch_train_loss)

#     print(f"Average Val Accuracy : {(100 * accuracy):.2f}")
#     print(f"Average Loss: {(epoch_train_loss):.2f}")