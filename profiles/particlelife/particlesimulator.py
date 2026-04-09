import torch
import cv2
import numpy as np
from rlhfalife.utils import Simulator, Generator, Rewarder
from typing import List, Any
import os


@torch.jit.script
def _simulate_particle_life_jit(
    steps: int,
    px: torch.Tensor,
    py: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    M_full: torch.Tensor,
    dt: float,
    beta: float,
    max_rad_sq: float,
    force_mult: float,
    friction_factor: float,
) -> torch.Tensor:
    N = px.size(0)
    history = torch.zeros(
        (steps, N, 2), dtype=torch.float32, device=torch.device("cpu")
    )

    for step in range(steps):
        dx = px.unsqueeze(0) - px.unsqueeze(1)
        dy = py.unsqueeze(0) - py.unsqueeze(1)

        dx = dx - torch.round(dx)
        dy = dy - torch.round(dy)

        d_sq = (dx * dx + dy * dy) / max_rad_sq

        mask = (d_sq < 1.0) & (d_sq > 1e-16)

        d_mask = torch.sqrt(d_sq[mask])

        f = torch.zeros_like(d_mask)

        repulsion_mask = d_mask < beta
        f[repulsion_mask] = d_mask[repulsion_mask] / beta - 1.0

        attraction_mask = d_mask >= beta
        if attraction_mask.any():
            i_idx, j_idx = torch.where(mask)
            M = M_full[i_idx[attraction_mask], j_idx[attraction_mask]]
            f[attraction_mask] = M * (
                1.0
                - torch.abs(2.0 * d_mask[attraction_mask] - 1.0 - beta) / (1.0 - beta)
            )

        force_mag = f / d_mask * force_mult

        fx = dx[mask] * force_mag
        fy = dy[mask] * force_mag

        i_idx, _ = torch.where(mask)

        total_fx = torch.zeros(N, dtype=torch.float32, device=px.device)
        total_fy = torch.zeros(N, dtype=torch.float32, device=px.device)

        total_fx.index_add_(0, i_idx, fx)
        total_fy.index_add_(0, i_idx, fy)

        vx = vx * friction_factor + total_fx * dt
        vy = vy * friction_factor + total_fy * dt

        px = px + vx * dt
        py = py + vy * dt

        px = px - torch.floor(px)
        py = py - torch.floor(py)

        history[step, :, 0] = px.cpu()
        history[step, :, 1] = py.cpu()

    return history


class ParticleSimulator(Simulator):
    """
    GPU Accelerated Particle Life Simulator.
    Simulates particles moving in a continuous space based on distance-dependent forces.

    Using a GPU for N particles: O(N^2) vectorized force computation.
    """

    def __init__(
        self,
        generator: Generator,
        size: tuple[int, int],
        particles_number: int,
        steps: int,
        friction_half_time: float,
        beta: float,
        max_radius: float,
        dt: float,
        force_factor: float,
        video_frames_sample_rate: int = 1,
        device: str = "cuda",
    ):
        """
        Initialize the simulator.

        Args:
            generator: the generator providing the simulation parameters.
            steps: number of simulation steps.
            dt: time step size.
            beta: core repulsion radius as fraction of max_radius.
            max_radius: maximum interaction radius.
            force_factor: scalar multiplied with forces.
            friction_half_time: half velocity decay time.
            device: 'cuda' or 'cpu'.
        """
        super().__init__(generator)
        self.steps = steps
        self.dt = dt
        self.beta = beta
        self.max_radius = max_radius
        self.force_factor = force_factor
        self.friction_factor = 0.5 ** (self.dt / friction_half_time)
        self.size = size
        self.particles_number = particles_number
        self.video_frames_sample_rate = video_frames_sample_rate
        self.device = device

        # Color table for video generation
        self.colors = torch.tensor(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [0, 255, 255],
                [255, 0, 255],
                [255, 128, 0],
                [255, 0, 128],
                [0, 255, 128],
                [128, 255, 0],
            ],
            dtype=torch.uint8,
            device=self.device,
        )

    def run(self, params_list: List[Any]) -> List[Any]:
        """
        Run simulations given parameters.
        Force law:
            If d < beta: f = d/beta - 1
            If beta < d < 1: f = interaction[t1, t2] * (1 - abs(2*d - 1 - beta) / (1 - beta))

        Args:
            params_list: List of parameters (interaction_matrix, positions, types).

        Returns:
            The outputs: List of dicts with 'types' and 'positions'
        """
        outputs = []
        for params in params_list:
            interaction_matrix, positions, types = params

            # Move to device if needed
            interaction_matrix = interaction_matrix.to(self.device, dtype=torch.float32)
            positions = positions.to(self.device, dtype=torch.float32)
            types = types.to(self.device, dtype=torch.long)

            N = positions.shape[0]
            vx = torch.zeros(N, dtype=torch.float32, device=self.device)
            vy = torch.zeros(N, dtype=torch.float32, device=self.device)

            px = positions[:, 0].clone()
            py = positions[:, 1].clone()

            M_full = interaction_matrix[types.unsqueeze(1), types.unsqueeze(0)]
            max_rad_sq = float(self.max_radius**2)
            force_mult = float(self.max_radius * self.force_factor)
            beta_f = float(self.beta)
            dt_f = float(self.dt)
            friction_f = float(self.friction_factor)

            history = _simulate_particle_life_jit(
                self.steps,
                px,
                py,
                vx,
                vy,
                M_full,
                dt_f,
                beta_f,
                max_rad_sq,
                force_mult,
                friction_f,
            )

            outputs.append({"types": types.cpu(), "positions": history})

        return outputs

    def save_output(self, output: Any, path: str) -> str:
        """
        Save the output to a file as a dictionary.
        """
        torch.save(output, f"{path}.pt")
        return f"{path}.pt"

    def load_output(self, path: str) -> Any:
        return torch.load(path)

    def save_param(self, param: Any, path: str) -> None:
        torch.save(param, f"{path}.pt")

    def load_param(self, path: str) -> Any:
        return torch.load(path)

    def save_video_from_output(self, output: Any, path: str) -> None:
        """
        Generate an mp4 video.
        """
        history = output["positions"]  # (T, N, 2)
        types = output["types"]  # (N,)
        T, N, _ = history.shape
        # Use config's size, defaulting to 500
        video_size = self.size[0] if isinstance(self.size, (list, tuple)) else self.size

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(path, fourcc, 30.0, (video_size, video_size))

        # Vectorized numpy indexing setup to skip slow cv2.circle loops
        pos = (history.numpy() * video_size).astype(np.int32)
        t_np = types.numpy()

        from .utils import get_color_tables

        colors_table = get_color_tables(max(10, types.max().item() + 1))

        particle_colors = colors_table.numpy()[t_np % len(colors_table)]
        particle_colors = particle_colors[:, ::-1]  # BGR

        # Circle radius 3 pixels
        offsets = []
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dx * dx + dy * dy <= 9:
                    offsets.append((dx, dy))

        for step in range(0, T, self.video_frames_sample_rate):
            frame = np.zeros((video_size, video_size, 3), dtype=np.uint8)
            p = pos[step]

            x = np.clip(p[:, 0], 3, video_size - 4)
            y = np.clip(p[:, 1], 3, video_size - 4)

            for dx, dy in offsets:
                frame[y + dy, x + dx] = particle_colors

            out_video.write(frame)

        out_video.release()
