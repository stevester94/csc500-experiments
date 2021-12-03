from math import floor
from torch.utils.data import DataLoader
from typing import Tuple
import torch
from torch import nn, optim

# BEGIN KEEP IN THIS ORDER
# It's critical that you override this function first before importing the rest of easyfsl
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
# END KEEP IN THIS ORDER


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
        log_update_frequency = 100

        all_loss = []
        self.train()

        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
        ) in enumerate(train_loader):
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
                print(f"[{episode_index} / {len(train_loader)}], Average Train Loss {sliding_average(all_loss, log_update_frequency):.2f}")

            # Validation
            if val_loader:
                if (episode_index + 1) % validation_frequency == 0:
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
            for _, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in enumerate(data_loader):
                correct, total = self.evaluate_on_one_task(
                    support_images, support_labels, query_images, query_labels
                )

                total_predictions += total
                correct_predictions += correct

        return correct_predictions / total_predictions

def split_ds_into_episodes(
    ds,
    n_way,
    n_shot,
    n_query,
    n_train_tasks,
    n_val_tasks,
    n_test_tasks,
    seed,
):
    import copy 

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
                n_way=n_way,
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

    # Deep copy necessary because of how tensors are retrieved from workers
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