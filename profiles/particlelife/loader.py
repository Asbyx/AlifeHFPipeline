import os
import torch
import numpy as np
from tqdm import tqdm
from rlhfalife.utils import Generator, Rewarder, Simulator, Loader
from .particlerewarder import ParticleRewarder
from .randomparticlegenerator import RandomFixedSizeParticleGenerator, RandomParticleGenerator
from .particlesimulator import ParticleSimulator
from typing import Tuple

class Loader(Loader):
    """
    Loader class for ParticleLife profile.
    Loads the Generator, Rewarder, and Simulator components.
    """
    
    def load(self, out_paths: dict, config: dict) -> Tuple[Generator, Rewarder, Simulator]:
        """
        Load the generator, rewarder and simulator.
        
        Args:
            out_paths: Dictionary containing paths:
                - 'outputs': path to the outputs folder
                - 'videos': path to the videos folder
                - 'params': path to the params folder
                - 'rewarder': path to the rewarders folder
                - 'generator': path to the generators folder
                - 'saved_simulations': path to the saved simulations folder
            config: Dictionary containing the config of the experiment:
                - size: size of the simulation
                - steps: number of steps in the simulation
                - particles_number: number of particles in the simulation
                - beta: beta parameter
                - force_factor: force factor parameter
                - max_radius: max radius parameter
                - friction_half_time: friction half time parameter
                - dt: dt parameter
        Returns:
            Tuple of (Generator, Rewarder, Simulator) instances
        """        
        device = config.get('device', 'cpu')
        print(f"\nUsing device: {device}")

        # Generator
        generator = RandomParticleGenerator(
            config['particles_number'], device
            )
    
        # Simulator
        simulator = ParticleSimulator(
            generator,
            config['size'], 
            config['particles_number'], 
            config['steps'], 
            config['friction_half_time'], 
            config['beta'], 
            config['max_radius'], 
            config['dt'], 
            config['force_factor'],
            config['video_frames_sample_rate']
            )

        # Rewarder
        rewarder_path = os.path.join(out_paths['rewarder'], config['model_name'] + '.pt')
        rewarder = ParticleRewarder(
            clipvip_path="./profiles/particlelife/rewarder/clipvip_32.pt",
            config=config,
            model_path=rewarder_path,
            device=device,
            simulator=simulator
        )
        if os.path.exists(rewarder_path):
            rewarder = rewarder.load()
        else:
            print(f"No model found at {rewarder_path}, using initialized model.")
        

        # --- simple pipeline test ---
        # outputs = simulator.run(generator.generate(5))
        # rewards = rewarder.rank(outputs)
        # print(rewards)
        # exit(0)

        #--- test video generation & preprocessing ---
        # import cv2
        # outputs = simulator.run(generator.generate(1))
        # simulator.save_video_from_output(outputs[0], "test.mp4")
        # preprocessed = rewarder.preprocess(outputs).cpu().numpy() # (B, T, C, H, W)
        # print(f"preprocessed shape: {preprocessed.shape}")
        
        # # Ensure the frames are in the right format for OpenCV (B, T, H, W, C) with values in 0-255 range
        # frames_for_video = np.transpose(preprocessed, (0, 1, 3, 4, 2))[0]
        
        # # Write frames to video
        # height, width = preprocessed.shape[3:]
        # video = cv2.VideoWriter("test_preprocessed.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
        # for i, frame in enumerate(frames_for_video):
        #     # Print color information for the first frame to debug
        #     if i == 0:
        #         # Calculate some statistics about the colors
        #         print(f"frame shape: {frame.shape}")
              
        #         # Count non-black pixels
        #         non_black = np.sum(np.any(frame > 0, axis=2))
        #         total_pixels = frame.shape[0] * frame.shape[1]
        #         print(f"  Non-black pixels: {non_black}/{total_pixels} ({non_black/total_pixels:.2%})")
              
        #         # List unique colors in the frame
        #         pixels = frame.reshape(-1, 3)
        #         pixel_tuples = [tuple(pixel) for pixel in pixels]
        #         unique_colors = set(pixel_tuples)
        #         print(f"  Unique colors: {len(unique_colors)}")
        #         print("  Color list:")
        #         for color in sorted(unique_colors):
        #             print(f"    RGB: {color}")

        #         print(f"Positions of non-black pixels:")
        #         for i in range(frame.shape[0]):
        #             for j in range(frame.shape[1]):
        #                 if np.any(frame[i, j] > 0):
        #                     print(f"  ({i}, {j})")
        #     video.write(frame)
        # video.release()
        # exit(0)
        

        #--- Benchmark preprocessing time ---
        # import time
        # start_time = time.time()
        # outputs = [simulator.load_output("out/particlelife/final_n/outputs/-49562264884308504")]
        # rewarder.preprocess(outputs).cpu().numpy() # (B, T, C, H, W)
        # end_time = time.time()
        # print(f"Preprocessing time: {end_time - start_time} seconds")

        # exit(0)
        return generator, rewarder, simulator

    def custom_script(self, generator: "Generator", rewarder: "Rewarder", simulator: "Simulator") -> None:
        """
        Custom script to run after the generator, rewarder and simulator are loaded.

        You can use this function to do any custom script, like testing, using your functions, print values,etc.
        """
        nb_iterations = 5
        nb_samples = 20

        generator = RandomFixedSizeParticleGenerator(
            types_number=5, particles_number=generator.particles_number, device=generator.device
        )

        population = generator.generate(nb_samples)
        
        for i in tqdm(range(nb_iterations)):
            outputs = simulator.run(population)
            rewards = torch.tensor(rewarder.rank(outputs)) # output a list

            # top 30% survive and reproduce (with mutation)
            top_k = int(0.3 * nb_samples)
            best_indices = torch.argsort(rewards, dim=0, descending=True)[:top_k]
            population = [population[i] for i in best_indices]
            simulator.save_video_from_output(outputs[best_indices[0]], f"C:\\Users\\elrbe\\all_code\\perso\\AlifeHub\\evo\\best_of_gen_{i}.mp4")

            mutated_population = RandomFixedSizeParticleGenerator.mutate(population)
            population = population + mutated_population
            
            # Rest is filled with new random individuals to maintain population size
            population = population + generator.generate(nb_samples - len(population))

        # produce video of top 30%
        outputs = simulator.run(population)
        rewards = torch.tensor(rewarder.rank(outputs)) 
        top_k = int(0.5 * nb_samples)
        top_indices = torch.argsort(rewards, dim=0, descending=True)[:top_k]
        for i in top_indices:
            simulator.save_video_from_output(outputs[i], f"C:\\Users\\elrbe\\all_code\\perso\\AlifeHub\\evo\\evolved_{i}_score_{rewards[i]:.2f}.mp4")
        exit(0)
