from typing import List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .simulator import Simulator
    from .rewarder import Rewarder

class Generator:
    """
    Abstract Generator class for generating parameters for an alife simulation
    Parameters are anything that can be used to configure the simulation.

    For example, for a cellular automaton, the parameters could be the initial state of the grid, or the neighborhood function.
    """

    #-------- To implement --------#
    def generate(self, nb_params: int) -> List[Any]:
        """
        Generate some parameters for the simulation.
        
        Args:
            nb_params: Number of different parameters to generate

        Returns:
            A list of parameters, of length nb_params. The parameters themselfs can be anything that can be converted to a string (override the __str__ method if needed).
        """
        raise NotImplementedError("Generator.generate must be implemented in inheriting class")

    def train(self, simulator: "Simulator", rewarder: "Rewarder") -> None:
        """
        Train the generator using the rewarder

        Args:
            simulator: Simulator for which the generator is trained
            rewarder: Rewarder to train with
        """
        raise NotImplementedError("Generator.train must be implemented in inheriting class")
    
    def save(self) -> None:
        """
        Save the generator to the path.

        Note: the path is expected to be specified by the user. This function will be called as such: generator.save()
        For example, one can define the path in the constructor of the inheriting class and use it here.
        """
        raise NotImplementedError("Generator.save must be implemented in inheriting class")

    def load(self) -> "Generator":
        """
        Load the generator from the path.
        
        Note: the path is expected to be specified by the user. This function will be called as such: generator.load()
        For example, one can define the path in the constructor of the inheriting class and use it here.

        Returns:
            The loaded generator
        """
        raise NotImplementedError("Generator.load must be implemented in inheriting class")

    #-------- Built in --------#
    def hash_params(self, params: List[Any]) -> List[int]:
        """
        Hash a list of parameters.
        
        Note: the parameters are expected to be hashable. If not, this function is expected to be overriden in the inheriting class.

        Args:
            params: List of parameters to hash

        Returns:
            A list of hashes of the parameters
        """
        try:
            return [hash(param) for param in params]
        except:
            raise Exception("Parameters have an unhashable type. Please override the hash_params method in the Generator class.") 