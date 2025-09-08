import torch
from textwrap import dedent


class Automaton:
    """
    Class that internalizes the rules and evolution of
    an Alife model. By default, the world tensor has shape
    (3,H,W) and should contain floats with values in [0.,1.].
    """

    def __init__(self, size):
        """
        Parameters :
        size : 2-uple (H,W)
            Shape of the CA world
        """
        self.h, self.w = size
        self.size = size

        self._worldmap = torch.zeros((3, self.h, self.w), dtype=float)  # (3,H,W), contains a 2D 'view' of the CA world
        x = torch.arange(0, self.w, dtype=float)
        y = torch.arange(0, self.h, dtype=float)
        self.Y, self.X = torch.meshgrid(y, x, indexing='ij') # (H,W)

    def step(self):
        return NotImplementedError('Please subclass "Automaton" class, and define self.step')

    def draw(self):
        """
        This method should be overriden. It should update the self._worldmap tensor,
        drawing the current state of the CA world. self._worldmap is a torch tensor of shape (3,H,W).
        If you choose to use another format, you should override the worldmap property as well.
        """
        return NotImplementedError('Please subclass "Automaton" class, and define self.draw')
    

    @property
    def worldmap(self):
        """
        Converts self._worldmap to cpu and returns it.

        Returns:
            torch.Tensor: (3,H,W) tensor
        """
        return self._worldmap.cpu()



    def get_help(self):
        doc = self.__doc__
        process = self.process_event.__doc__
        if(doc is None):
            doc = "No description available"
        if(process is None):
            process = "No interactivity help available"
        return dedent(doc), dedent(process)
