from typing import List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_managers import TrainingDataset

class Rewarder:
    """
    Abstract Rewarder class for estimating the reward
    It is expected to be trained on a dataset of pairs of simulations, with the winner of each pair, and be used to train a Generator.
    """

    #-------- To implement --------#
    def rank(self, data: List[Any]) -> List[float]:
        """
        Given a list of outputs of a Simulator, this function is expected to return a list of scores, one for each output.
        The function name is "rank" for historical reasons, but "score" would be more appropriate.
        
        Args:
            data: Data to rank, array-like of outputs of a Simulator.

        Returns:
            An array-like of the same length as data, where the i-th element is the score for the i-th sample.
        """
        raise NotImplementedError("Rewarder.rank must be implemented in inheriting class")

    def train(self, dataset: "TrainingDataset") -> None:
        """
        Train the rewarder on the ranked pairs.  
        
        The dataset delivers triplets of (path_to_output_1, path_to_output_2, winner). Usage example:
        ```
        for path_to_output_1, path_to_output_2, winner in dataset:
            rewarder.rank([path_to_output_1, path_to_output_2])
            ...
        ```

        Args:
            dataset: TrainingDataset instance containing the dataset.
        """
        raise NotImplementedError("Rewarder.train must be implemented in inheriting class")

    def save(self) -> None:
        """
        Save the rewarder.
        
        Note: the path is expected to be specified by the user. This function will be called as such: rewarder.save()
        For example, one can define the path in the constructor of the inheriting class and use it here.
        """
        raise NotImplementedError("Rewarder.save must be implemented in inheriting class")

    def load(self) -> "Rewarder":
        """
        Load the rewarder.

        Note: the path is expected to be specified by the user. This function will be called as such: rewarder.load()
        For example, one can define the path in the constructor of the inheriting class and use it here.

        Returns:
            The loaded rewarder
        """
        raise NotImplementedError("Rewarder.load must be implemented in inheriting class") 