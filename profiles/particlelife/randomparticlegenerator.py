from tqdm import tqdm

from rlhfalife.utils import *
import torch
from typing import List, Any

class RandomParticleGenerator(Generator):
    def __init__(self, particles_number: int, gen_mode: str = 'random', device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.particles_number = particles_number
        self.gen_mode = gen_mode
        self.device = device
        self.rewarder = None
        self.simulator = None

    def _random_gen(self, nb_params: int) -> List[Any]:
        types_number = torch.randint(1, 10, (nb_params,)).tolist()
        return [
            (torch.randn((types_number[i], types_number[i]), device=self.device),
            torch.rand((self.particles_number, 2), device=self.device),
            torch.randint(0, types_number[i], (self.particles_number,), device=self.device))
            for i in range(nb_params)
        ]

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
        if self.gen_mode == 'random':
            return self._random_gen(nb_params)
        
        elif self.gen_mode == 'filtering':
            if self.simulator is None or self.rewarder is None:
                print("Simulator or Rewarder not provided. Falling back to random generation.")
                return self._random_gen(nb_params)
            
            print(f"Generating {nb_params * 3} candidates...")
            params = self._random_gen(nb_params * 3)
            print("Running simulations...")
            outputs = self.simulator.run(params)
            print("Rewarding...")
            ranks = self.rewarder.rank(outputs)
            
            ranks = torch.argsort(torch.tensor(ranks), dim=0, descending=True)
            return [params[i.item()] for i in ranks[:nb_params]]
            
        elif self.gen_mode == 'evolution':
            print("Evolution mode selected. Running evolutionary experiment...")
            if self.simulator is None or self.rewarder is None:
                print("Simulator or Rewarder not provided. Falling back to random generation.")
                return self._random_gen(nb_params)
                
            num_generations = 3
            pop_size = max(nb_params * 2, 20)
            num_parents = max(pop_size // 2, 2)
            num_elite = max(num_parents // 2, 1)

            population = self._random_gen(pop_size)
            
            for gen in tqdm(range(num_generations), desc="Evolutionary Generations"):
                tqdm.write(f"Running simulation")
                outputs = self.simulator.run(population)
                tqdm.write(f"Calculating rewards")
                rewards = torch.tensor(self.rewarder.rank(outputs))
                
                tqdm.write(f"Evolution step")
                if gen == num_generations - 1:
                    print("Final generation reached. Returning top individuals.")
                    topk_indices = torch.topk(rewards, nb_params).indices
                    return [population[i.item()] for i in topk_indices]
                
                top_indices = torch.topk(rewards, num_parents).indices
                parents = [population[i.item()] for i in top_indices]
                
                next_gen = parents[:num_elite]
                mutation_rate = 0.2
                
                while len(next_gen) < pop_size:
                    parent_idx = torch.randint(0, num_parents, (1,)).item()
                    parent = parents[parent_idx]
                    
                    # Mutate attraction matrix
                    attraction_matrix, positions, types = parent
                    if torch.rand(1).item() < mutation_rate:
                        noise = torch.randn_like(attraction_matrix) * 0.1
                        mutated_attraction_matrix = torch.clamp(attraction_matrix + noise, -1, 1)
                    else:
                        mutated_attraction_matrix = attraction_matrix.clone()
                        
                    next_gen.append((mutated_attraction_matrix, positions, types))
                    
                population = next_gen
                
            return self._random_gen(nb_params) # Fallback

    def set_rewarder(self, rewarder: Rewarder) -> None:
        self.rewarder = rewarder

    def train(self, simulator: Simulator, rewarder: Rewarder) -> None:
        self.simulator = simulator
        self.rewarder = rewarder
        print("Generator trained (references to simulator and rewarder saved).")

    def save(self) -> None:
        pass
    
    def load(self) -> None:
        raise NotImplementedError("Not available for this generator.")

class RandomFixedSizeParticleGenerator(RandomParticleGenerator):
    def __init__(self, types_number: int, particles_number: int, gen_mode: str = 'random', device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(particles_number, gen_mode, device)
        self.types_number = types_number

    def _random_gen(self, nb_params: int) -> List[Any]:
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
        return [
            (torch.randn((self.types_number, self.types_number), device=self.device),
            torch.rand((self.particles_number, 2), device=self.device),
            torch.randint(0, self.types_number, (self.particles_number,), device=self.device))
            for i in range(nb_params)
        ]
    
    @staticmethod
    def mutate(params):
        """
        Mutate the attraction matrix of the parameters by adding a small random noise.
        """
        mutated_params = []
        for param in params:
            attraction_matrix, positions, types = param
            noise = torch.randn_like(attraction_matrix) * 0.1
            mutated_attraction_matrix = torch.clamp(attraction_matrix + noise, -1, 1)
            mutated_params.append((mutated_attraction_matrix, positions, types))
        return mutated_params