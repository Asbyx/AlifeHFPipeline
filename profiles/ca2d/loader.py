from typing import TYPE_CHECKING, Tuple
from .generator import CA2DGenerator
from ..lenia.rewarder import LeniaRewarder
from .simulator import CA2DSimulator
from pathlib import Path

class Loader:
    """
    Loader class to load a Generator, a Rewarder and a Simulator
    """
    def load(self, out_paths: dict, config: dict) -> Tuple["CA2DGenerator", "LeniaRewarder", "CA2DSimulator"]:
        """
        Load the generator, rewarder and simulator for the CA2D profile

        Args:
            out_paths: Dictionary containing: 
                - 'outputs': path to the outputs folder,
                - 'videos': path to the videos folder,
                - 'params': path to the params folder,
                - 'rewarder': path to the rewarders folder,
                - 'generator': path to the generators folder,
                - 'saved_simulations': path to the saved simulations folder,
            config: Dictionary containing the config of the experiment
        """

        device = config['device']
        print("Initializing generator...")
        generator = CA2DGenerator(**config["generator"], device=device)
        print("Initializing simulator...")
        simulator = CA2DSimulator(generator=generator,**config["simulator"], device=device)
        print("Initializing rewarder...")
        rewarder_path = Path(out_paths['rewarder']) / f"{config['name']}.pt"
        rewarder = LeniaRewarder(simulator=simulator,model_path=rewarder_path,**config["rewarder"], device=device)

        if rewarder_path.exists():
            print("Loading existing rewarder model...")
            rewarder.load()
            generator.train(simulator, rewarder)

        return generator, rewarder, simulator