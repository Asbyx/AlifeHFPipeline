import numpy as np
import torch
import colorsys # For HSL-like distinct colors

def get_color_tables(num_types: int) -> torch.Tensor:
    """
    Generates a table of distinct colors using HSL color space.
    Args:
        num_types: The number of distinct types/colors needed.
    Returns:
        torch.Tensor: A tensor of shape (num_types, 3) with BGR uint8 color values.
                    Returns an empty tensor if num_types is 0 or negative.
    """
    if num_types <= 0:
        return torch.empty((0, 3), dtype=torch.uint8)
        
    colors_bgr = []
    for i in range(num_types):
        hue = i / num_types
        lightness = 0.5 
        saturation = 0.9
        rgb_float = colorsys.hls_to_rgb(hue, lightness, saturation)
        bgr_color = [
            int(rgb_float[2] * 255), 
            int(rgb_float[1] * 255), 
            int(rgb_float[0] * 255)
        ]
        colors_bgr.append(bgr_color)
    return torch.tensor(colors_bgr, dtype=torch.uint8, device='cpu')

def render_frame(
    positions: torch.Tensor,  # (N, 2) on target device
    types: torch.Tensor,      # (N,) on target device, 0-indexed
    target_width: int,
    target_height: int,
    color_table: torch.Tensor, # (max_types, 3) on target device
    particle_radius: int = 3
) -> torch.Tensor: # (H, W, 3) on target device
    """
    Renders a single frame of particles as a torch.Tensor array using a vectorized approach.
    Particles are drawn in order (higher index on top if overlapping).
    Args:
        positions (torch.Tensor): Particle positions for a single frame, 
                                                shape (num_particles, 2), normalized [0,1].
        types (torch.Tensor): Particle types, shape (num_particles,). Indices for color_table.
        target_width (int): Width of the output frame.
        target_height (int): Height of the output frame.
        color_table (torch.Tensor): Color lookup table, shape (max_type_id + 1, 3), BGR uint8.
        particle_radius (int): Radius of the circle to draw for each particle.
    Returns:
        torch.Tensor: Rendered frame of shape (target_height, target_width, 3), dtype=torch.uint8, BGR format.
    """
    N = positions.shape[0]
    device = positions.device

    if N == 0:
        return torch.zeros((target_height, target_width, 3), dtype=torch.uint8, device=device)

    # 1. Calculate particle centers in pixel coordinates (float for precision with grid)
    # Normalized positions [0,1] are scaled to [0, W-1] and [0, H-1]
    x_coords_px_norm = positions[:, 0] * (target_width -1) # Using (target_width-1) for 0-W-1 range
    y_coords_px_norm = positions[:, 1] * (target_height-1) # Using (target_height-1) for 0-H-1 range

    # 2. Get particle colors (ensure types are long for indexing)
    particle_colors = color_table[types.long()]  # (N, 3)

    # 3. Create pixel grid (float for precise distance calculation)
    # Grid coordinates are 0 to H-1 and 0 to W-1
    y_grid, x_grid = torch.meshgrid(
        torch.arange(target_height, device=device, dtype=torch.float32),
        torch.arange(target_width, device=device, dtype=torch.float32),
        indexing='ij'
    )  # y_grid, x_grid are (H, W)

    # 4. Expand coordinate dimensions for broadcasting
    # Particle centers: (N, 1, 1)
    x_part_exp = x_coords_px_norm[:, None, None]
    y_part_exp = y_coords_px_norm[:, None, None]
    # Pixel grid: (1, H, W) -> no, keep as (H,W) and broadcast with (N,1,1)
    # x_grid_exp = x_grid[None, :, :]
    # y_grid_exp = y_grid[None, :, :]

    # 5. Compute squared distances from each particle center to each pixel grid point
    # dist_sq will be (N, H, W)
    dist_sq = (x_grid - x_part_exp)**2 + (y_grid - y_part_exp)**2
    
    radius_sq = float(particle_radius**2) # Ensure float for comparison

    # 6. Create masks: True where pixel is covered by particle
    particle_pixel_masks = dist_sq <= radius_sq  # (N, H, W), boolean

    # 7. Determine foremost particle for each pixel (painter's algorithm)
    # Values: 1 for particle 0, 2 for particle 1, ..., N for particle N-1.
    # 0 if not covered by any of these particles.
    particle_order_values = torch.arange(1, N + 1, device=device, dtype=torch.long) # (N,)
    
    # masked_values: (N, H, W). Contains particle_order_value if covered, else 0.
    masked_values = torch.where(
        particle_pixel_masks, 
        particle_order_values[:, None, None], # Expand (N,) to (N,1,1) to broadcast with (N,H,W) mask
        torch.tensor(0, device=device, dtype=torch.long)
    )

    # Find the maximum order value (i.e., an indirect way to get index of foremost particle)
    # max_order_val_map will be (H, W). Contains 0 if no particle, or k+1 if particle k is foremost.
    max_order_val_map, _ = torch.max(masked_values, dim=0) # (H, W)

    # Initialize frame to background (black)
    frame = torch.zeros((target_height, target_width, 3), dtype=torch.uint8, device=device)

    # Create a mask for pixels that are covered by at least one particle
    covered_pixels_mask = max_order_val_map > 0 # (H, W), boolean

    # Get the actual particle indices (0 to N-1) for covered pixels
    # max_order_val_map stored k+1, so subtract 1.
    # Only select from max_order_val_map where covered_pixels_mask is True.
    foremost_particle_indices_flat = max_order_val_map[covered_pixels_mask] - 1 # (num_covered_pixels,)

    # Get the colors for these foremost particles
    # types are used to index color_table. Here we need colors of particles specified by foremost_particle_indices_flat
    colors_to_apply = particle_colors[foremost_particle_indices_flat] # (num_covered_pixels, 3)

    # Apply colors to the frame
    frame[covered_pixels_mask] = colors_to_apply

    return frame