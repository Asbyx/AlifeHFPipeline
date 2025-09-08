from rlhfalife import Generator, Simulator, Rewarder
from .simulator import CA2DSimulator
from typing import List, Any
import random
from math import ceil


class CA2DGenerator(Generator):
    """
    Abstract Generator class for generating parameters for an alife simulation
    Parameters are anything that can be used to configure the simulation.

    For example, for a cellular automaton, the parameters could be the initial state of the grid, or the neighborhood function.
    """

    def __init__(self, seed=None, device='cpu', latest_rewarder=None, latest_simulator=None):
        """
        Initialize the generator for CA2D parameters

        Args:
            seed: Seed for the random number generator, for reproducibility.
            device: Device to run the simulation on. Defaults to "cpu". If "cuda" is available, it will be used.
        """
        super().__init__()
        self.seed = seed
        if(seed is not None):
            random.seed(seed)
        self.device = device
        self.latest_rewarder = None
        self.latest_simulator = None

    #-------- To implement --------#

    def generato(self, nb_params: int) -> List[Any]:
        """
        Generate some parameters for the simulation.
        
        Args:
            nb_params: Number of different parameters to generate

        Returns:
            A list of parameters, of length nb_params. The parameters themselfs can be anything that can be converted to a string (override the __str__ method if needed).
        """
        sampled_params = []

        for _ in range(nb_params):
            s_num = random.randint(0, 2**9-1)
            b_num = random.randint(0, 2**9-1)
            sampled_params.append((s_num, b_num))
    
        return sampled_params

    def generate(self, nb_params: int, threshold: float = 0.2) -> List[Any]:
        """
            Generate parameters, keeping only a fraction threshold of the best ones

            Args:
                nb_params: Number of different parameters to generate
                threshold: ]0.,1] float, fraction of the best parameters to keep

            Returns:
                A list of nb_params selected parameters
        """
        selected_params = []
        if(self.latest_rewarder is not None and self.latest_simulator is not None):
            num_sets_needed = ceil(1/threshold)
            for _ in range(num_sets_needed):
                params = self.generato(nb_params)
                # Simulate and evaluate them
                scores = []

                sim_result = self.latest_simulator.run(params) # list of (T,C,H,W)
                preprocessed = self.latest_rewarder.preprocess(sim_result) # (B,T',3,H,W)
                scores = self.latest_rewarder(preprocessed) # (B, )

                # Select the top x% of parameters, and add to the selected params
                threshold_index = ceil(len(scores) * threshold)
                selected_params.extend([p for p, s in sorted(zip(params, scores), key=lambda x: x[1], reverse=True)[:threshold_index]])
                if(len(selected_params)>nb_params):
                    break # break if we have enough params
            
            return selected_params[:nb_params]
        else:
            return self.generato(nb_params)


    def train(self, simulator: "CA2DSimulator", rewarder: "Rewarder") -> None:
        """
        'Trains' the generator : stores the latest rewarder and simulator.
        They are using during 'generate_threshold', by generating parameters, 
        simulating them, and keeping only the top x% of scores.

        Args:
            simulator: Simulator used to generate parameters.
            rewarder: Trained rewarder
        """
        self.latest_rewarder : "Rewarder" = rewarder
        self.latest_simulator : "CA2DSimulator" = simulator

    def save(self) -> None:
        """
        Save the generator to the path
        """
        print('Saving not implemented yet, but it would essentially just be saving the rewarder')

    def load(self) -> "Generator":
        """
        Load the generator from the path

        Returns:
            The loaded generator
        """
        print('Loading not implemented yet, but it would essentially just be loading the rewarder')