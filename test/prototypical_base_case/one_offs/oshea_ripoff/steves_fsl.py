from math import floor
from numpy.core.shape_base import vstack
from torch.utils.data import DataLoader
from typing import Tuple
import torch
from torch import nn, optim
import sys,time

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
        self.best_validation_avg_loss = float("inf")

    def fit(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        val_loader: DataLoader = None,
        validation_frequency: int = 1000,
        log_frequency=100
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

        all_loss = []
        self.train()

        examples_processed = 0
        interval_start_time = time.time()

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


            examples_processed += support_images.shape[0]
            examples_processed += query_images.shape[0]

            # Log training loss in real time
            if episode_index % log_frequency == 0:
                print(f"[{episode_index} / {len(train_loader)}], Average Train Loss {sliding_average(all_loss, log_frequency):.2f}, Examples per second: {examples_processed / (time.time() - interval_start_time):.2f}")
                sys.stdout.flush()
                examples_processed = 0
                interval_start_time = time.time()
            # # Validation
            # if val_loader:
            #     if (episode_index + 1) % validation_frequency == 0:
            #         val_acc, val_loss = self.validate(val_loader)

            #         train_loss_history.append(sliding_average(all_loss, log_update_frequency))
            #         val_loss_history.append(val_loss)

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
        validation_accuracy, validation_avg_loss = self.evaluate(val_loader)
        # print(f"Val Accuracy: {(100 * validation_accuracy):.2f}%, Val Avg Loss: {validation_avg_loss:.2f}")
        # # If this was the best validation performance, we save the model state
        # if validation_avg_loss < self.best_validation_avg_loss:
        #     print("Best so far")
        #     self.best_model_state = self.state_dict()
        #     self.best_validation_avg_loss = validation_avg_loss

        return validation_accuracy, validation_avg_loss

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
        total_loss = 0

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
                correct, total, loss = self.evaluate_on_one_task(
                    support_images, support_labels, query_images, query_labels
                )

                total_predictions += total
                correct_predictions += correct
                total_loss += loss


        return correct_predictions / total_predictions, (total_loss/len(data_loader)).detach().item()

    def evaluate_on_one_task(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> [int, int, float]:
        """
        Returns the number of correct predictions of query labels, and the total number of
        predictions.
        """
        self.process_support_set(support_images.cuda(), support_labels.cuda())

        classification_scores = self(query_images.cuda())
        loss = self.compute_loss(classification_scores, query_labels.cuda())


        return (
            torch.max(
                classification_scores,
                1,
            )[1]
            == query_labels.cuda()
        ).sum().item(), len(query_labels), loss.detach().data



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