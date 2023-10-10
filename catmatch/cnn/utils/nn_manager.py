import collections
import pathlib
import time
from typing import Any, Callable, TypeVar

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

import catmatch.cnn.utils.nn_utils as utils


def compute_loss_and_accuracy(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_criterion: torch.nn.modules.loss._Loss,
    modify_model_output: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    try_gpu: bool = True,
):
    """
    Computes the average loss and the accuracy over the whole dataset
    in dataloader.
    Args:
        dataloder: Validation/Test dataloader
        model: torch.nn.Module
        loss_criterion: The loss criterion, e.g: torch.nn.CrossEntropyLoss()
    Returns:
        [average_loss, accuracy]: both scalar.
    """
    average_loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            # Transfer images/labels to GPU VRAM, if possible
            if try_gpu:
                X_batch = utils.to_cuda(X_batch)
                Y_batch = utils.to_cuda(Y_batch)
            # Forward pass the images through our model
            out = model(X_batch)
            output_probs = modify_model_output(out)
            average_loss += loss_criterion(output_probs, Y_batch)
            accuracy += (output_probs.argmax(dim=1) == Y_batch).float().sum() / len(
                Y_batch
            )
        average_loss /= len(dataloader)
        accuracy /= len(dataloader)
    return average_loss.cpu().float(), accuracy.cpu().float()  # type: ignore


TModel = TypeVar("TModel", bound=torch.nn.Module)


# TODO: Add support for early stopping using validation dataset
class NNManager:
    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        early_stop_count: int,
        epochs: int,
        model: TModel,
        dataloaders: tuple[DataLoader, DataLoader, DataLoader],
        call_model: Callable[[TModel, Any], Any] = lambda model, x: model(x),
        modify_model_output: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        try_gpu: bool = True,
    ):
        """
        Initialize our trainer class.
        Dataloader is a tuple of the dataloader for train, val and test.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.epochs = epochs

        # Since we are doing multi-class classification, we use CrossEntropyLoss
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        # Initialize the model
        self.model: torch.nn.Module = model
        # Transfer model to GPU VRAM, if possible.
        self.model = utils.to_cuda(self.model) if try_gpu else self.model
        self.modify_model_output = modify_model_output
        self.call_model = call_model
        self.try_gpu = try_gpu
        print(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)  # type: ignore # noqa

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = dataloaders

        # Validate our model everytime we pass through 50% of the dataset
        self.num_steps_per_val = len(self.dataloader_train) // 2
        self.global_step = 0
        self.start_time = time.time()

        # Tracking variables
        self.train_history: dict = dict(
            loss=collections.OrderedDict(), accuracy=collections.OrderedDict()
        )
        self.validation_history: dict = dict(
            loss=collections.OrderedDict(), accuracy=collections.OrderedDict()
        )
        self.checkpoint_dir = pathlib.Path("checkpoints")

    def validation_step(self):
        """
        Computes the loss/accuracy for all three datasets.
        Train, validation and test.
        """
        self.model.eval()  # type: ignore
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val,
            self.model,
            self.loss_criterion,
            self.modify_model_output,
        )
        self.validation_history["loss"][self.global_step] = validation_loss
        self.validation_history["accuracy"][self.global_step] = validation_acc
        used_time = time.time() - self.start_time
        print(
            f"Epoch: {self.epoch:>1}",
            f"Batches per seconds: {self.global_step / used_time:.2f}",
            f"Global step: {self.global_step:>6}",
            f"Validation Loss: {validation_loss:.2f}",
            f"Validation Accuracy: {validation_acc:.3f}",
            sep=", ",
        )
        self.model.train()  # type: ignore

    def should_early_stop(self):
        """
        Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss
        # list.
        val_loss = self.validation_history["loss"]
        if len(val_loss) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = list(val_loss.values())[-self.early_stop_count :]
        first_loss = relevant_loss[0]
        if first_loss == min(relevant_loss):
            print("Early stop criteria met")
            return True
        return False

    def train_step(self, X_batch, Y_batch):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the
        train step.
        The function returns the mean loss value which is then automatically logged in
        our variable self.train_history.
        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
        # Y_batch is the CIFAR10 image label. Shape: [batch_size]
        # Transfer images / labels to GPU VRAM, if possible
        if self.try_gpu:
            X_batch = utils.to_cuda(X_batch)
            Y_batch = utils.to_cuda(Y_batch)

        # Perform the forward pass
        out = self.call_model(self.model, X_batch)
        predictions = self.modify_model_output(out)
        # Compute the cross entropy loss for the batch
        loss = self.loss_criterion(predictions, Y_batch)
        # Backpropagation
        loss.backward()
        # Gradient descent step
        self.optimizer.step()
        # Reset all computed gradients to 0
        self.optimizer.zero_grad()

        return loss.detach().cpu().item()

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """

        def should_validate_model():
            return self.global_step % self.num_steps_per_val == 0

        should_early_stop = False
        for epoch in range(self.epochs):
            print(f"---- Epoch {epoch + 1} of {self.epochs} ----")
            self.epoch = epoch
            # Perform a full pass through all the training samples
            for X_batch, Y_batch in tqdm(self.dataloader_train):
                loss = self.train_step(X_batch, Y_batch)
                self.train_history["loss"][self.global_step] = loss
                self.global_step += 1
                # Compute loss/accuracy for validation set
                if should_validate_model():
                    self.validation_step()
                    self.save_model()
                    if self.should_early_stop():
                        print("Early stopping.")
                        should_early_stop = True
                        break
            if should_early_stop:
                break
        self._create_training_plots()

    def save_model(self):
        def is_best_model():
            """
            Returns True if current model has the lowest validation loss
            """
            val_loss = self.validation_history["loss"]
            validation_losses = list(val_loss.values())
            return validation_losses[-1] == min(validation_losses)

        state_dict = self.model.state_dict()  # type: ignore
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")

        utils.save_checkpoint(state_dict, filepath, is_best_model())

    def load_best_model(self):
        state_dict = utils.load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}"  # type: ignore # noqa
            )
            return
        self.model.load_state_dict(state_dict)  # type: ignore

    def test(self) -> float:
        """Test the model using the test dataset and generates a classification report
        and returns the accuracy score.

        Returns:
            float: The accuracy of the model.
        """
        y_pred = []
        y_true = []
        self.model.eval()
        with torch.no_grad():
            for X_batch, Y_batch in tqdm(self.dataloader_test):
                if self.try_gpu:
                    X_batch = utils.to_cuda(X_batch)
                    Y_batch = utils.to_cuda(Y_batch)
                out = self.call_model(self.model, X_batch)  # type: ignore
                output = self.modify_model_output(out)
                pred = output.argmax(
                    dim=1
                )  # get the index of the max log-probability  # noqa
                y_pred.append(pred)
                y_true.append(Y_batch)
        y_pred = torch.cat(y_pred).cpu().numpy()
        y_true = torch.cat(y_true).cpu().numpy()
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        return float(accuracy_score(y_true, y_pred))

    def _create_training_plots(self):
        # Save plots and show them
        plt.figure(figsize=(20, 8))
        plt.subplot(1, 2, 1)
        plt.title("Cross Entropy Loss")
        utils.plot_loss(
            self.train_history["loss"], label="Training loss", npoints_to_average=10
        )
        utils.plot_loss(self.validation_history["loss"], label="Validation loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        utils.plot_loss(
            self.validation_history["accuracy"], label="Validation Accuracy"
        )
        plt.legend()
        plt.show()
        plt.show()
