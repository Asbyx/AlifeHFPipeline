from rlhfalife import Simulator, Generator
from typing import List, Any, Tuple
import torch
from .ca2d import CA2D
from showtens import save_video
from pathlib import Path
import json


class CA2DSimulator(Simulator):
    """
        Simulator class for 2-dimensional Cellular Automata
    """
    def __init__(self, generator: "Generator", world_size=(200,200),dot_init=False, run_length: int = 600, num_output_frames=12, device='cpu' ):
        """
        Initialize the simulator with a generator.

        Args:
            generator: Generator to use for generating parameters
            world_size: Size of the world to simulate
            dot_init: If True, initialize the world with a 4*4 random dot in the center. Otherwise, initalize with a noise square.
            run_length: Number of frames to run the simulation for
            num_output_frames: Number of frames to keep in output tensor. Spaced Linearly
            device: Device to run the simulation on. Defaults to "cpu". If "cuda" is available, it will be used.
        """
        self.generator = generator
        self.world_size = world_size
        self.run_length = run_length
        self.device = device
        self.num_output_frames = num_output_frames

        self.automaton = CA2D(size=world_size, random=not dot_init,device=device)


    #-------- To implement --------#
    def run(self, params: List[Tuple[int]]) -> List[torch.Tensor]:
        """
        Run the simulation with the given parameters, and return outputs.
        Will return only even frames to avoid blinking.

        Args:
            params: Parameter list of ca2d in integer notation.

        Returns:
            The outputs of the simulation (List of (T,C,H,W) tensors)
        """
        outputs = []
        for param in params:
            self.automaton.change_num(s_num=param[0], b_num=param[1])  # update the parameters of the simulation
            self.automaton.reset()
            output = []
            for s in range(self.run_length):
                self.automaton.step()
                if(s%2==0):
                    #  Append only only even frames, to avoid blinking
                    self.automaton.draw()
                    output.append(self.automaton.worldmap) # List of (3,H,W) tensors
            outputs.append(torch.stack(output,dim=0))

        return outputs  # List of (T,C,H,W) tensors

    def save_output(self, output: Any, path: str) -> str:
        """
        Save the output to the path.

        Args:
            output: Output to save
            path: Path to the file to save the output to, not including the extension

        Returns:
            The path to the saved output
        """
        torch.save(self._process_output(output), path+'.pt')

        return path + '.pt'    

    def _process_output(self, tensvid):
        """
            Given a video tensor, returns the processed tensor.

            Args:
            tensvid : (T,3,H,W) representing the videos

            Returns:
            (T',3,H',W') tensor, processed tensvid in model's format
        """

        T,C,H,W = tensvid.shape
        assert self.num_output_frames <= T, f'tensvid {T} frames, need at least {self.num_output_frames} frames'

        # Take self.num_output_frames equally spaced frames
        tensvid = tensvid[torch.linspace(0,T-1,self.num_output_frames).long()] # (T',3,H,W)

        return tensvid

    def save_video_from_output(self, output: Any, path: str) -> None:
        """
        Convert the output to a video and save it at path.
        It is important that the video is a .mp4 file !


        Args:
            output: Output to save
            path: Path to the file to save the video to

        Example usage:
            save_video_from_output(output, "out/profile/videos/video_number1.mp4")
        """
        save_video(output, folder=Path(path).parent, name=Path(path).stem, fps=30)

    def save_param(self, param: Tuple[int], path: str) -> str:
        """
        Save the param to the path.
        
        Args:
            param: ca2d parameter in integer notation
            path: Path to the file to save the parameter to, without extension
        """
        param = {
            "s_num": param[0],
            "b_num": param[1]
        }
        with open(path+'.json', 'w') as f:
            json.dump(param, f)
        return path + '.json'

    def load_output(self, path: str) -> Any:
        """
        Load the output from the path.

        Args:
            path: Path to the file to load the output from. The path is the path to the file to load the output from. No need to specify the extension.

        Returns:
            The loaded output
        """
        
        return torch.load(path+'.pt', map_location=self.device)  # (T,C,H,W) tensor
    
    def load_param(self, path: str) -> Any:
        """
        Load the param from the path.

        Args:
            path: Path to the file to load the parameter from. The path is the path to the file to load the parameter from. No need to specify the extension.

        Returns:
            The loaded parameter
        """    
        with open(path+'.json', 'r') as f:
            param = json.load(f)
        return (param["s_num"], param["b_num"])
