from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .generator import Generator
    from .rewarder import Rewarder
    from .simulator import Simulator

class Loader:
    """
    Abstract Loader class to load a Generator, a Rewarder and a Simulator
    Any subclass of this class must be named Loader.
    """
    def load(self, out_paths: dict, config: dict) -> Tuple["Generator", "Rewarder", "Simulator"]:
        """
        Load the generator, rewarder and simulator

        Args:
            out_paths: Dictionary containing: 
                - 'outputs': path to the outputs folder,
                - 'videos': path to the videos folder,
                - 'params': path to the params folder,
                - 'rewarder': path to the rewarders folder,
                - 'generator': path to the generators folder,
                - 'saved_simulations': path to the saved simulations folder,
            config: Dictionary containing the config of the experiment
        """
        raise NotImplementedError("Loader.load must be implemented in inheriting class") 