import argparse

import gymnasium as gym
import numpy as np

import gym_nes_ium  # noqa
from gymnasium.utils.play import play
import itertools


def _get_args():
    """Parse arguments from the command line and return them."""
    parser = argparse.ArgumentParser(description=__doc__)
    # add the argument for the Super Mario Bros environment to run
    parser.add_argument(
        "--rom",
        "-r",
        type=str,
        help="The path to the ROM to play.",
        required=True,
    )
    # add the argument for the mode of execution as either human or random
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="human",
        choices=["human", "random"],
        help="The execution mode for the emulation.",
    )
    # add the argument for the number of steps to take in random mode
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=5_000,
        help="The number of random steps to take.",
    )
    return parser.parse_args()


def get_keys_to_action():
    """Return the dictionary of keyboard keys to actions."""
    # keyboard keys in an array ordered by their byte order in the bitmap
    # i.e. right = 7, left = 6, ..., B = 1, A = 0
    buttons = np.array(
        [
            ord("d"),  # right
            ord("a"),  # left
            ord("s"),  # down
            ord("w"),  # up
            ord("\r"),  # start
            ord(" "),  # select
            ord("p"),  # B
            ord("o"),  # A
        ]
    )
    # the dictionary of key presses to controller codes
    keys_to_action = {}
    # the combination map of values for the controller
    values = 8 * [[0, 1]]
    # iterate over all the combinations
    for combination in itertools.product(*values):
        # unpack the tuple of bits into an integer
        byte = int("".join(map(str, combination)), 2)
        # unwrap the pressed buttons based on the bitmap
        pressed = buttons[list(map(bool, combination))]
        # assign the pressed buttons to the output byte
        keys_to_action[tuple(sorted(pressed))] = byte

    return keys_to_action


# get arguments from the command line
args = _get_args()

play(gym.make("NES/SuperMarioBros-v0"), keys_to_action=get_keys_to_action(), noop=0)
