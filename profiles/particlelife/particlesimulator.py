from rlhfalife.utils import *
import torch
import json
import cv2
import numpy as np
from .utils import get_color_tables, render_frame
from typing import List, Any

class ParticleSimulator(Simulator):
    """
    ParticleSimulator is a class that simulates the particle life automaton.
    
    Parameters:
        torch.Tensor, the attraction matrix. Size is (types_number, types_number).
        torch.Tensor, the positions of the particles. Size is (particles_number, 2).
        torch.Tensor, the types of the particles. Size is (particles_number,).
    """
    def __init__(self, generator: Generator, size: tuple[int, int], particles_number: int, steps: int, friction_half_time: float, beta: float, max_radius: int, dt: float, force_factor: float, video_frames_sample_rate: int = 1):
        """
        Initialize the ParticleSimulator.

        Args:
            generator: The generator to use for the simulation.
            friction_half_time: The half-time of the friction.
            beta: The beta parameter.
            max_radius: The maximum radius of the particles.
            dt: The time step of the simulation.
            force_factor: The force factor of the simulation.
            sample_rate: Save every Nth step (default=1). Higher values use less memory but record fewer frames.
        """
        super().__init__(generator)

        self.friction_factor = 0.5**(dt/friction_half_time)
        self.beta = beta
        self.max_radius = max_radius
        self.dt = dt
        self.force_factor = force_factor
        self.device = generator.device
        self.steps = steps
        self.size = torch.tensor(size, device=self.device)
        self.particles_number = particles_number
        self.video_frames_sample_rate = video_frames_sample_rate

    def run(self, params: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Run the simulation with the given parameters.

        Args:
            params: The parameters to use for the simulation.
                attraction_matrix (torch.Tensor): The attraction matrix.
                positions (torch.Tensor): The positions of the particles.
                types (torch.Tensor): The types of the particles.
        """
        outputs = []
        for param in params:
            # Ensure input tensors are on the correct device
            attraction_matrix = param[0].to(self.device)
            positions = param[1].to(self.device)
            types = param[2].to(self.device)
            velocities = torch.zeros((self.particles_number, 2), dtype=torch.float32, device=self.device)
            
            # Store positions history on GPU during simulation
            positions_history = torch.zeros((self.steps, self.particles_number, 2), dtype=torch.float32, device=self.device)
            
            for step in range(self.steps):            
                normalized_distances = torch.norm(positions[:, None, :] - positions[None, :, :], dim=2)/self.max_radius
                directions = -(positions[:, None, :] - positions[None, :, :]) / normalized_distances[:, :, None]
                directions = directions.nan_to_num(0.0)

                F = torch.where(
                    normalized_distances < self.beta,
                    normalized_distances/self.beta - 1,
                    torch.where(
                        normalized_distances < 1,
                        attraction_matrix[types[:, None], types[None, :]] * (1 - torch.abs(2*normalized_distances - 1 - self.beta) / (1-self.beta)),
                        torch.zeros_like(normalized_distances, dtype=torch.float32, device=self.device)
                    )
                )

                F = F[:, :, None] * directions
                F = F.sum(dim=1)*self.force_factor*self.max_radius

                velocities = velocities*self.friction_factor + F*self.dt

                positions += velocities*self.dt                
                positions = torch.fmod(positions, 1.0)
                positions = torch.where(positions < 0, positions + 1.0, positions)

                # Save current positions to history
                positions_history[step] = positions.clone()

            # Transfer all data to CPU at the end
            positions_history_cpu = positions_history.cpu()

            output = {
                "types": types.cpu(),
                "positions": positions_history_cpu,
            }
            
            outputs.append(output)
        return outputs

    def save_output(self, output: torch.Tensor, path: str) -> str:
        """
        Save the output to the path.

        Args:
            output: The output to save.
            path: The path to save the output to.

        Returns:
            The path to the saved output.
        """
        # Save as torch tensors instead of JSON
        torch.save({
            "types": output["types"],
            "positions": output["positions"]
        }, f"{path}.pt")
        return f"{path}.pt"

    def load_output(self, path: str) -> torch.Tensor:
        return torch.load(f"{path}.pt")
    
    def save_video_from_output(self, output: torch.Tensor, path: str) -> None:
        """
        Save the video from the output using the utility rendering function.

        Args:
            output: The output to save, containing positions and types.
            path: The path to save the video to.
        """
        positions_history = output["positions"]  # This is already on CPU from the run method
        types_cpu = output["types"] # This is already on CPU
        
        # Move necessary tensors to the device
        types = types_cpu.to(self.device)
        positions_history_device = positions_history.to(self.device)

        sim_size_cpu = self.size.cpu().numpy() if isinstance(self.size, torch.Tensor) else self.size
        width, height = int(sim_size_cpu[0]), int(sim_size_cpu[1])
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(path, fourcc, fps, (width, height))
        
        color_table_size = types.max().item() + 1
        # get_color_tables returns a CPU tensor, move it to device
        color_table = get_color_tables(color_table_size).to(self.device)
        
        # Filter frames based on sample rate
        indices = torch.arange(0, len(positions_history_device), self.video_frames_sample_rate, device=self.device)
        sampled_positions = positions_history_device[indices]
        
        num_frames_to_render = len(sampled_positions)
        if num_frames_to_render == 0:
            video.release()
            return

        # Pre-allocate tensor for all frames on the GPU
        all_frames_gpu = torch.zeros((num_frames_to_render, height, width, 3), dtype=torch.uint8, device=self.device)
        
        for i, positions_for_frame in enumerate(sampled_positions):
            # render_frame expects positions for a single frame
            frame = render_frame(
                positions_for_frame, 
                types, 
                width, 
                height, 
                color_table, 
                particle_radius=3 # Ensure particle_radius is an int
            )
            all_frames_gpu[i] = frame
        
        # Transfer all frames to CPU at once
        all_frames_numpy = all_frames_gpu.cpu().numpy()
        
        # Write frames to video
        for frame_np in all_frames_numpy:
            video.write(frame_np)
        
        video.release()

    def save_param(self, param: Any, path: str) -> str:
        """
        Save the param to the path.
        
        Args:
            param: The param to save.
            path: The path to save the param to.

        Returns:
            The path to the saved param.
        """
        torch.save(param, f"{path}.pt")
        return f"{path}.pt"
    
    def load_param(self, path: str) -> Any:
        """
        Load the param from the path.

        Args:
            path: The path to load the param from.

        Returns:
            The loaded param.
        """
        return torch.load(path)