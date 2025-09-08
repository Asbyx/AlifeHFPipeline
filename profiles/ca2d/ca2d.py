from .automaton import Automaton
import torch
import torch.nn.functional as F
import colorsys


class CA2D(Automaton):
    """
    2D Cellular Automaton, with outer-holistic rules and binary values.
    The state of a pixel in the next generation depends on its own state and
    the sum of the values of its neighbors.
    """

    def __init__(self, size, s_num="23", b_num="3", random=False, highlight_color=None,device="cpu"):
        """
        Params :
        size : tuple, size of the automaton
        s_num : str, rule for survival
        b_num : str, rule for birth
        random : bool, if True, the initial state of the automaton is random. Otherwise, a small random square is placed in the middle.
        highlight_color : 3-uple of [0.,1.] floats, color of the highlight
        """
        super().__init__(size)

        self.s_num = self.get_num_from_rule(s_num)  # Translate string to number form
        self.b_num = self.get_num_from_rule(b_num)  # Translate string to number form
        self.random = random
        self.device = device

        self.world = torch.zeros((self.h, self.w), dtype=torch.int, device=device)
        self.reset()

        self.kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], device=device, dtype=torch.float)[
            None, None, :, :
        ]  # (1,1,3,3) as required by Pytorch

        # self.change_highlight_color()
        if(highlight_color is not None):
            self.highlight_color = torch.tensor(highlight_color, device=device)
        else:
            self.highlight_color = torch.tensor([150.,81.,111.])/255.
        self.highlight_color = self.highlight_color.to(device)

        self.decay_speed = 0.8

        self._worldmap = self._worldmap.to(device)

    def change_highlight_color(self):
        """
        Changes the highlight color, gets random hue
        """
        hue = torch.rand(1).item()
        light = 0.5
        saturation = 0.5
        self.highlight_color = torch.tensor(
            colorsys.hls_to_rgb(hue, saturation, light), dtype=torch.float, device=self.device
        )

    def get_num_from_rule(self, num : str):
        """
        Get the rule number for the automaton, given the string representation of the rule.
        """
        rule_out = 0
        if num != "":  # If the num is empty, the associated number is 0
            rule_out = sum([2 ** int(d) for d in num])

        return rule_out

    def get_rule_from_num(self, num : int):
        """
            Get the rule string from the rule number
        """
        numbers_list = [int(num//2**i % 2) for i in range(9)]

        out = ""
        for i, n in enumerate(numbers_list):
            if n == 1:
                out += str(i)

        return out

    def reset(self):
        """
        Resets the automaton to the initial state.
        """
        self._worldmap = torch.zeros((3, self.h, self.w), device=self.device)

        if self.random:
            self.world = self.get_init_mat(0.3)
        else:
            self.world = torch.zeros_like(self.world, dtype=torch.int, device=self.device)
            self.world[self.h // 2 - 1 : self.h // 2 + 1, self.w // 2 - 1 : self.w // 2 + 1] = torch.randint(
                0, 2, (2, 2), device=self.device
            )

    def draw(self):
        """
        Updates the worldmap with the current state of the automaton.
        """
        echo = torch.clamp(self._worldmap - self.decay_speed * (1-self.highlight_color[:, None, None]), min=0, max=1).to(
            self.device
        )
        self._worldmap = torch.clamp(
            self.world[None, :, :].expand(3, -1, -1).to(dtype=torch.float) + echo, min=0, max=1
        ).to(self.device)

    def change_num(self, s_num: int | str, b_num: int | str):
        """
        	Changes the rule of the automaton to the one specified by s_num and b_num
        """
        if(isinstance(s_num, str)):
            s_num = self.get_num_from_rule(s_num)
        if(isinstance(b_num, str)):
            b_num = self.get_num_from_rule(b_num)
            
        self.s_num = s_num
        self.b_num = b_num
        self.reset()

    def step(self):
        # Generate tensors for all 8 neighbors
        w, e = self.world.roll(-1, 0), self.world.roll(1, 0)
        n, s = self.world.roll(-1, 1), self.world.roll(1, 1)
        sw, se = w.roll(1, 1), e.roll(1, 1)
        nw, ne = w.roll(-1, 1), e.roll(-1, 1)

        count = w + e + n + s + sw + se + nw + ne

        # count2 = self.convolve_count(self.world)
        # assert torch.all(count==count2)

        self.world = torch.where(
            self.world == 1, self.get_nth_bit(self.s_num, count), self.get_nth_bit(self.b_num, count)
        ).to(torch.int)

    def convolve_count(self, world):
        """
        Convolves the world with the kernel to get the count of neighbors
        """
        pad_world = F.pad(
            world[None, None, :, :].to(torch.float), (1, 1, 1, 1), mode="circular"
        )  # (1,1,H+2,W+2) circular padded

        return F.conv2d(pad_world, self.kernel)[0, 0, :, :].to(torch.int)  # (H,W)

    def get_nth_bit(self, num, s):
        """
        Get the nth bit of the number num
        """
        return (num >> s) & 1

    def get_init_mat(self, rand_portion):
        """
        Get initialization matrix for CA

        Params :
        rand_portion : float, portion of the screen filled with noise.
        """
        batched_size = torch.tensor([self.h, self.w])
        randsize = (batched_size * rand_portion).to(dtype=torch.int16)  # size of the random square
        randstarts = (batched_size * (1 - rand_portion) / 2).to(
            dtype=torch.int16
        )  # Where to start the index for the random square

        randsquare = torch.where(torch.randn(*randsize.tolist()) > 0, 1, 0)  # Creates random square of 0s and 1s

        init_mat = torch.zeros((self.h, self.w), dtype=torch.int16)
        init_mat[randstarts[0] : randstarts[0] + randsize[0], randstarts[1] : randstarts[1] + randsize[1]] = randsquare
        init_mat = init_mat.to(torch.int16)

        return init_mat.to(self.device)  # (B,H,W)

