from rlhfalife.utils import *
import torch
from typing import List, Any

class RandomParticleGenerator(Generator):
    def __init__(self, particles_number: int, filtering_mode: bool = False, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.particles_number = particles_number
        self.filtering_mode = filtering_mode
        self.device = device

    def generate(self, nb_params: int) -> List[Any]:
        """
        Generate a list of parameters for the particle life automaton.

        Args:
            nb_params: The number of parameters to generate.

        Returns:
            A list of parameters.
            params:
                torch.Tensor, the attraction matrix. Size is (types_number, types_number).
                torch.Tensor, the positions of the particles. Size is (particles_number, 2).
                torch.Tensor, the types of the particles. Size is (particles_number,).
        """
        types_number = torch.randint(1, 10, (nb_params,)).tolist()
        if not (self.filtering_mode and hasattr(self, 'rewarder')):
            return [
                (torch.randn((types_number[i], types_number[i]), device=self.device),
                torch.rand((self.particles_number, 2), device=self.device),
                torch.randint(0, types_number[i], (self.particles_number,), device=self.device))
                for i in range(nb_params)
            ]

        params = [
            (torch.randn((types_number[i%nb_params], types_number[i%nb_params]), device=self.device),
            torch.randn((self.particles_number, 2), device=self.device),
            torch.randint(0, types_number[i%nb_params], (self.particles_number,), device=self.device))
            for i in range(nb_params*10)
        ]
        ranks = self.rewarder.rank(params)
        ranks = torch.argsort(ranks, dim=0, descending=True)
        params = [params[i] for i in ranks[:nb_params]]
        return params

    def set_rewarder(self, rewarder: Rewarder) -> None:
        self.rewarder = rewarder

    def train(self, simulator: Simulator, rewarder: Rewarder) -> None:
        pass

    def save(self) -> None:
        pass
    
    def load(self) -> None:
        raise NotImplementedError("Not available for this generator.")

