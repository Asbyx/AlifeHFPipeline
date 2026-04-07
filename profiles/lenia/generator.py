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
                gen_mode : 'default', 'random', 'filtering' or 'evolution'. 'default' will generate a fixed set of parameters, 'random' will generate random parameters, 'filtering' will generate a large batch of random parameters and keep the best ones according to the rewarder, 'evolution' will use a simple evolutionary algorithm to evolve the parameters according to the rewarder.
                k_size : size of the kernel
        """
        super().__init__()
        self.gen_mode = gen_mode
        assert gen_mode in ['default', 'random', 'filtering', 'evolution'], f"Invalid generation mode: {gen_mode}"

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
            case "evolution":
                if self.rewarder is None:
                    raise ValueError("No rewarder found for evolution generation mode.")
                elif not self.rewarder.loaded:
                    print("Rewarder not loaded yet, generating random parameters instead.")
                    dataparams = LeniaParams.default_gen(batch_size=nb_params, k_size=self.k_size, device=self.device)
                    return [dataparams[i] for i in range(dataparams.batch_size)]
                
                num_generations = 5
                pop_size = max(nb_params * 2, 20)
                num_parents = max(pop_size // 2, 2)
                num_elite = max(num_parents // 2, 1)


                mutation_rate = 0.2
                mutation_magnitude = 0.1
                
                # Initialize random population
                population = [LeniaParams.default_gen(batch_size=1, k_size=self.k_size, device=self.device) for _ in range(pop_size)]
                
                for gen in range(num_generations):
                    sims = self.simulator.run(population)
                    rewards = self.rewarder.rank(sims)
                    
                    del sims
                    gc.collect()
                    
                    rewards_tensor = torch.tensor(rewards)
                    
                    if gen == num_generations - 1:
                        # Final generation: select the raw top k for the return type
                        topk_indices = torch.topk(rewards_tensor, nb_params).indices
                        return [population[i.item()] for i in topk_indices]
                    
                    # Selection
                    top_indices = torch.topk(rewards_tensor, num_parents).indices
                    parents = [population[i.item()] for i in top_indices]
                    
                    # Next generation list
                    next_gen = []
                    
                    # Elitism
                    next_gen.extend(parents[:num_elite])
                    
                    # Reproduction (mutation)
                    while len(next_gen) < pop_size:
                        parent_idx = torch.randint(0, num_parents, (1,)).item()
                        parent = parents[parent_idx]
                        child = parent.mutate(magnitude=mutation_magnitude, rate=mutation_rate)
                        next_gen.append(child)
                        
                    population = next_gen
                    del rewards_tensor, parents, next_gen, top_indices
                    gc.collect()
                    
                return [population[i] for i in range(min(nb_params, len(population)))]
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