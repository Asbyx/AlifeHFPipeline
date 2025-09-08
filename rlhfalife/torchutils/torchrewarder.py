from ..utils import Rewarder, Simulator
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from ..data_managers import TrainingDataset
import hashlib
from typing import List, Any, Optional
import wandb
from ..benchmarker import test_rewarder_on_benchmark
from tqdm import tqdm
from pathlib import Path

class TorchRewarder(nn.Module, Rewarder):
    """
    A rewarder that uses a torch model. 

    The only method that needs to be implemented is the forward method.

    Note: using this class assume that the outputs are torch tensors (dtype=torch.float32) saved as pt files, with torch.save.
    """
    def __init__(self, config: dict, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", simulator: Optional[Simulator] = None, wandb_params: dict = None):
        """
        Initialize the TorchRewarder.
    
        Args:
            config: Dictionary containing configuration parameters. All are optional.
                lr (default 0.001): Learning rate
                epochs (default 100): Number of epochs to train
                batch_size (default 16): Batch size
                val_split (default 0.2): Validation split
                early_stopping_patience (default 10): Early stopping patience
                loss (default "cross_entropy"): Loss function to use. Can be "margin" or "cross_entropy".
                test_set_path (default None): Path to the test set.
            model_path: Path to save or load the model
            device: Device to run the model on. Defaults to "cuda" if available, otherwise "cpu".
            simulator: Optional Simulator instance, required for test set evaluation.
            wandb_params: Dictionary containing wandb parameters. Defaults to None.
        """
        nn.Module.__init__(self)
        self.config = config or {}
        self.device = device
        self.optimizer = None
        self.lr_scheduler = None

        self.model_path = model_path
        self.wandb_params = wandb_params
        self.simulator = simulator

        self.loss = {
            "margin": lambda scores1, scores2, y: F.margin_ranking_loss(scores1, scores2, y*2-1, margin=0.1), # convert to {-1, 1}
            "cross_entropy": lambda scores1, scores2, y: F.cross_entropy(torch.stack([scores1, scores2],dim=1), y.long(), reduction='mean')
        }[self.config.get('loss', 'cross_entropy')]

        if self.config.get('test_set_path', None) is not None:
            self.test_set = {"benchmark": self.config['test_set_path']}
            if self.simulator:
                print(f"Using test set from {self.config['test_set_path']}")
            else:
                print(f"Test set path provided ({self.config['test_set_path']}), but Simulator instance is missing. Test set evaluation will be skipped.")
        else:
            print("No test set provided. To create a test set, first create a benchmark and save it in a custom path, then provide the path to the test set in the config under the 'test_set_path' key.")
            self.test_set = None

    #-------- Methods to implement --------#
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor, assumed to be torch tensors already on the correct device. Should be a batch of data.
            
        Returns:
            Output tensor, shape (batch_size)
        """
        raise NotImplementedError("Forward method must be implemented by child classes")
    
    def preprocess(self, data: List[Any]) -> torch.Tensor:
        """
        Preprocess the data.

        Args:
            data: List of data to preprocess.
            
        Returns:
            Preprocessed data ready for forward pass (B, *)
        """
        return data
    
    #-------- End of methods to implement --------#

    def _setup_optimizer(self):
        """Set up the optimizer using the config parameters"""
        lr = self.config.get('lr', 0.001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def _setup_lr_scheduler(self):
        """
            Set up the learning rate scheduler using the config parameters.
            Uses a linear warmup by default.
        """
        warmup = self.config.get('warmup', True) # Use linear warmup by default
        if warmup:
            self.lr_scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-5, end_factor=1.0, total_iters=100)
        else:
            self.lr_scheduler = optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0) # Do nothing, basically

    def rank(self, data):
        """
        Rank the data based on the model's predictions.
        
        Args:
            data: Data to rank; shape (batch_size, *)
            
        Returns:
            List of float rewards for each input
        """
        data = self.preprocess(data)

        # Make the data a torch tensor
        if isinstance(data, torch.Tensor):
            data = data.to(self.device)
        elif isinstance(data, np.ndarray):
            data = torch.FloatTensor(data).to(self.device)
        elif isinstance(data, list):
            data = torch.stack(data).to(self.device)
        else:
            raise ValueError("Data must be a numpy array or a list")

        super().train(False)
        with torch.no_grad():
            scores = self(data)
            
        return scores.cpu().numpy().tolist()
    
    def _get_batches(self, dataset: TrainingDataset, indices, batch_size):
        """
        Get batches from the dataset using specified indices.
        
        Args:
            dataset: The dataset to sample from.
            indices: The indices to use for sampling.
            batch_size: Size of each batch.
            
        Yields:
            Batches of data.
        """
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_data = [dataset[j] for j in batch_indices]
            
            # Unpack the batch data
            path1s, path2s, winners = zip(*batch_data)
            
            # Load and preprocess the data using caching
            data1 = torch.stack([self._load_or_preprocess(path) for path in path1s])
            data2 = torch.stack([self._load_or_preprocess(path) for path in path2s])
            
            if not isinstance(data1[0], torch.Tensor):
                raise ValueError("The outputs are not saved as torch tensors. Please save the outputs as torch tensors using torch.save. Note: numpy arrays saved with torch.save will still be saved as numpy arrays.")

            yield data1, data2, torch.tensor(winners, dtype=torch.float32, device=self.device)
    
    def _evaluate_dataset(self, dataset_batches):
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset_batches: Generator yielding batches of data
            
        Returns:
            Average loss and accuracy over the dataset
        """
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        batch_size = self.config.get('batch_size', 16)
        
        super().train(False)
        with torch.no_grad():
            for batch_data1, batch_data2, batch_winners in dataset_batches:
                # Get predictions
                scores1 = self(batch_data1) # (B,)
                scores2 = self(batch_data2) # (B,)

                # Compute loss
                loss = self.loss(scores1, scores2, batch_winners)
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = (scores1 < scores2).int()
                mask_non_tie = (batch_winners==0) | (batch_winners==1) # ignore ties in accuracy
                correct += (predictions[mask_non_tie] == batch_winners[mask_non_tie]).sum().item()

                total += batch_winners[mask_non_tie].shape[0]
                
                batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy

    def _get_train_val_indices(self, dataset: TrainingDataset, config: dict) -> tuple:
        """
        Get training and validation indices from the dataset.
        
        Args:
            dataset: The dataset to split
            config: The config to use for the split
            
        Returns:
            tuple: (train_indices, val_indices), indices of the dataset
        """
        unique_simulations = dataset.get_simulations_hashes()
        
        val_split = config.get('val_split', 0.2)
        val_size = int(len(unique_simulations) * val_split)
        val_sims = unique_simulations[:val_size]

        val_indices = [i for i, (h1, h2, _) in enumerate(dataset) if any(sim in h1 for sim in val_sims) or any(sim in h2 for sim in val_sims)]
        train_indices = [i for i in range(len(dataset)) if i not in val_indices]
        return train_indices, val_indices
    
    def _train_single_batch(self, batch_data1, batch_data2, batch_winners):
        """
        Train on a single batch of data.
        
        Args:
            batch_data1: First batch of data
            batch_data2: Second batch of data
            batch_winners: Batch of winner labels
            
        Returns:
            tuple: (loss, correct_predictions, total_samples)
        """        
        # Get predictions
        scores1 = self(batch_data1)
        scores2 = self(batch_data2)
        
        # Compute loss
        loss = self.loss(scores1, scores2, batch_winners)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        # Calculate metrics
        predictions = (scores1 < scores2).int()
        correct = (predictions == batch_winners).sum().item()
        
        return loss.item(), correct, len(batch_winners)

    def _train_single_epoch(self, train_indices: list, dataset: TrainingDataset, batch_size: int):
        """
        Train for a single epoch.
        
        Args:
            train_indices: List of training indices
            dataset: Training dataset
            batch_size: Batch size
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
        # Shuffle training indices for each epoch
        random.shuffle(train_indices)
        
        # Get batches for training
        for batch_data1, batch_data2, batch_winners in self._get_batches(dataset, train_indices, batch_size):
            loss, batch_correct, batch_total = self._train_single_batch(batch_data1, batch_data2, batch_winners)
            
            total_loss += loss
            correct += batch_correct
            total += batch_total
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy

    def _log_progress(self, epoch: int, epochs: int,
                     train_loss: float, train_accuracy: float,
                     val_loss: float, val_accuracy: float,
                     test_pairwise_accuracy: Optional[float] = None):
        """
        Log training progress to console and wandb if configured.
        
        Args:
            epoch: Current epoch number
            epochs: Total number of epochs
            train_loss: Training loss
            train_accuracy: Training accuracy
            val_loss: Validation loss
            val_accuracy: Validation accuracy
            test_pairwise_accuracy: Test set pairwise accuracy (optional)
        """
        log_data = {
            "metrics/train_loss": train_loss,
            "metrics/train_accuracy": train_accuracy,
            "metrics/val_loss": val_loss,
            "metrics/val_accuracy": val_accuracy,
            "other/epoch": epoch,
            "other/lr": self.optimizer.param_groups[0]['lr'],
        }
        if test_pairwise_accuracy is not None and not np.isnan(test_pairwise_accuracy):
            log_data["test_pairwise_accuracy"] = test_pairwise_accuracy

        if self.wandb_params:
            self.wandb_run.log(log_data)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Mean Train Loss: {train_loss:.4f}, Mean Train Accuracy: {train_accuracy:.4f}")
            print(f"  Mean Val Loss: {val_loss:.4f}, Mean Val Accuracy: {val_accuracy:.4f}")
            if test_pairwise_accuracy is not None and not np.isnan(test_pairwise_accuracy):
                print(f"  Test Pairwise Accuracy: {test_pairwise_accuracy:.4f}")

    def train(self, dataset: TrainingDataset):
        """
        Train the model on the dataset of pairs using cross validation if specified.
        
        Args:
            dataset: TrainingDataset instance containing pairs of simulations with labels
            
        The config can contain:
            val_split (float, default 0.2): Ratio of SIMULATIONS to use for validation. Not pairs, it would rigg the validation set since the rewarder would have already seen some simulations before.
            early_stopping_patience (int, default 3): Number of epochs to wait for improvement before stopping.
            Other parameters remain the same as before.
        """
        if not hasattr(self, 'optimizer'):
            raise ValueError("Optimizer not set up. Call __init__ method at initialization of the child class.")
        
        self._setup_optimizer()
        self._setup_lr_scheduler()
    
        if self.wandb_params:
            self.wandb_run = wandb.init(**self.wandb_params)

        # Training parameters
        batch_size = self.config.get('batch_size', 16)
        epochs = self.config.get('epochs', 100)
        early_stopping_patience = self.config.get('early_stopping_patience', 3)

        dataset_size = len(dataset)
        train_indices, val_indices = self._get_train_val_indices(dataset, self.config)
        print(f"Training on {dataset_size} pairs, for {dataset.simulations_number} simulations. Number of training pairs: {len(train_indices)}, number of validation pairs: {len(val_indices)}.")
        
        # Track best model
        best_model_state = None
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in tqdm(range(epochs)):
            # Train for one epoch
            super().train()
            train_loss, train_accuracy = self._train_single_epoch(train_indices, dataset, batch_size)
            
            # Validation phase
            val_loss, val_accuracy = self._evaluate_dataset(
                self._get_batches(dataset, val_indices, batch_size)
            )
            
            # Test set evaluation
            test_pairwise_accuracy = float('nan') # Initialize with nan
            if self.test_set and self.simulator:
                # Ensure the rewarder is in eval mode for testing
                super().train(False)
                test_pairwise_accuracy = test_rewarder_on_benchmark(
                    simulator=self.simulator, 
                    rewarder=self, 
                    out_paths=self.test_set, 
                    verbose=False
                )

            # Log progress
            self._log_progress(epoch, epochs,
                                train_loss, train_accuracy,
                                val_loss, val_accuracy,
                                test_pairwise_accuracy) # Pass test_pairwise_accuracy
            
            # Early stopping and model checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
        # Load best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
        if self.wandb_params:
            self.wandb_run.finish()
    
    def save(self):
        """
        Save the model to disk using the model_path attribute.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load(self):
        """
        Load the model from disk using the model_path attribute.
        
        Returns:
            The loaded rewarder
        """
        if Path(self.model_path).exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Update config
            self.config = checkpoint.get('config', self.config)
            
            # Load model state
            self.load_state_dict(checkpoint['model_state_dict'])
            super().train(False)
            
            # Re-setup optimizer with possibly updated config
            self._setup_optimizer()
            
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"No model found at {self.model_path}, using initialized model")
            
        return self

    def _get_preprocessed_path(self, data_path: str) -> str:
        """
        Get the path where the preprocessed version of a file should be stored.
        
        Args:
            data_path: Original data file path
            
        Returns:
            Path where preprocessed version should be stored
        """
        # Create temp directory next to model path if it doesn't exist
        model_dir = Path(self.model_path).parent
        preprocessed_dir = model_dir / "preprocessed_temp"
        preprocessed_dir.mkdir(exist_ok=True)
        
        # Create unique filename based on original path and last modified time
        file_stat = Path(data_path).stat()
        unique_id = f"{data_path}_{file_stat.st_mtime}"
        filename_hash = hashlib.md5(unique_id.encode()).hexdigest()
        
        return str(preprocessed_dir / f"{filename_hash}.pt")
    
    def _load_or_preprocess(self, data_path: str) -> torch.Tensor:
        """
        Load preprocessed data if it exists, otherwise preprocess and save.
        
        Args:
            data_path: Path to the original data file
            
        Returns:
            Preprocessed data as torch tensor
        """
        preprocessed_path = self._get_preprocessed_path(data_path)
        
        # If preprocessed file exists and is newer than original, load it
        if Path(preprocessed_path).exists():
            if Path(preprocessed_path).stat().st_mtime >= Path(data_path).stat().st_mtime:
                return torch.load(preprocessed_path, map_location=self.device)
        
        # Otherwise preprocess and save
        data = torch.load(data_path, map_location=self.device)
        preprocessed_data = self.preprocess([data])[0]
        torch.save(preprocessed_data, preprocessed_path)
        return preprocessed_data
