import torch
import gc
from rlhfalife import Generator, Simulator, Rewarder
from typing import List, Any
from .utils.leniaparams import LeniaParams

class LeniaGenerator(Generator):

    def __init__(self, gen_mode:str, k_size=25, device='cpu'):
        """
            Temp Generator for Lenia. This will be replaced by a better generator later.

            Args:
                gen_mode : 'default', 'random' or 'ptf'
                k_size : size of the kernel
        """
        super().__init__()
        self.gen_mode = gen_mode
        assert gen_mode != 'ptf', "ptf mode not implemented yet"

        self.k_size = k_size
        self.device = device

        self.rewarder = None
        self.simulator = None

    def generate(self, nb_params: int) -> List[Any]:
        """
        Generate some parameters for the simulation.
        
        Args:
            nb_params: Number of different parameters to generate

        Returns:
            A list of parameters, of length nb_params. The parameters themselfs can be anything that can be converted to a string (override the __str__ method if needed).
        """
        match self.gen_mode:
            case 'default':
                batch_params = LeniaParams.default_gen(batch_size=nb_params, k_size=self.k_size, device=self.device)
                return [batch_params[i] for i in range(batch_params.batch_size)]
            case 'random':
                batch_params = LeniaParams.random_gen(batch_size=nb_params, k_size=self.k_size, device=self.device)
                return [batch_params[i] for i in range(batch_params.batch_size)]
            case 'filtering':
                if self.rewarder is None:
                    raise ValueError("No rewarder found for filtering generation mode.")
                elif not self.rewarder.loaded:
                    print("Rewarder not loaded yet, generating random parameters instead.")
                    dataparams = LeniaParams.default_gen(batch_size=nb_params, k_size=self.k_size, device=self.device)
                    return [dataparams[i] for i in range(dataparams.batch_size)]
                # Generate a large batch of random parameters, simulate them and keep the best ones according to the rewarder.
                large_batch_size = nb_params * 3
                
                batch_params = LeniaParams.default_gen(batch_size=large_batch_size, k_size=self.k_size, device=self.device)
                
                sims = self.simulator.run([batch_params[i] for i in range(large_batch_size)]) # (large_batch_size, num_frames, size, size, channels)
                rewards = self.rewarder.rank(sims) # (large_batch_size, )
                
                del sims  # Frees up memory from the large simulations tensor
                gc.collect()
                
                topk_indices = torch.topk(torch.tensor(rewards), nb_params).indices
                return [batch_params[i.item()] for i in topk_indices]
        raise ValueError(f"Invalid generation mode {self.gen_mode}")

    def train(self, simulator: "Simulator", rewarder: "Rewarder") -> None:
        """
        Train the generator using the rewarder

        Args:
            simulator: Simulator for which the generator is trained
            rewarder: Rewarder to train with
        """
        self.rewarder = rewarder
        self.simulator = simulator
        print("POOOOOF, generator trained !")
    
    def save(self) -> None:
        """
        Save the generator to the path
        """
        print("Fake saving of the generator")

    def load(self) -> "Generator":
        """
        Load the generator from the path

        Returns:
            The loaded generator
        """
        print(f"Fake loading of the generator.")