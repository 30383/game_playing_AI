import gymnasium as gym
import numpy as np

from abc import ABC, abstractmethod
from typing import Any, SupportsFloat

from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.spaces import Box, Discrete

from .simple_nes import simple_nes


class NESEnv(ABC, gym.Env):
    action_space = Discrete(256)
    """
    The Space object corresponding to valid actions, all valid actions should be
    contained with the space. It is a bitmap of button press values for the 8 NES
    buttons.
    """

    observation_space = Box(
        low=0, high=255, shape=simple_nes.screen_shape_24_bit, dtype=np.uint8
    )
    """
    The Space object corresponding to valid observations, all valid observations should
    be contained with the space. It is static across all instances.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}
    """
    The metadata of the environment containing rendering modes, rendering fps, etc.
    """

    def __init__(self, rom: str, render_mode: str = None):
        self.render_mode = render_mode

        print(rom)

        self.rom = rom

        # setup a placeholder for a "human" render mode viewer
        self.viewer = None

        # initialize the C++ object for running the environment
        simple_nes.env = self.rom

        # setup the controllers, screen, and RAM buffers
        self.controllers = [simple_nes.controller(port) for port in range(2)]
        self.screen = simple_nes.screen()
        self.ram = simple_nes.memory()

    @property
    def observation(self) -> ObsType:
        return self.screen

    @property
    @abstractmethod
    def _reward(self) -> SupportsFloat:
        pass

    @property
    @abstractmethod
    def _terminated(self) -> bool:
        pass

    @property
    @abstractmethod
    def _truncated(self) -> bool:
        pass

    @property
    @abstractmethod
    def _info(self) -> dict[str, Any]:
        pass

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Run one timestep of the environment’s dynamics using the agent actions.

        When the end of an episode is reached (`terminated` or `truncated`), it is
        necessary to call `reset()` to reset this environment’s state for the next
        episode.

        Parameters:
        - action (ActType): an action provided by the agent to update the environment
        state.

        Returns:
        - observation (ObsType): An element of the environment’s `observation_space` as
        the next observation due to the agent actions. An example is a numpy array
        containing the positions and velocities of the pole in CartPole.

        - reward (SupportsFloat): The reward as a result of taking the action.

        - terminated (bool): Whether the agent reaches the terminal state (as defined
        under the MDP of the task) which can be positive or negative. An example is
        reaching the goal state or moving into the lava from the Sutton and Barton,
        Gridworld. If true, the user needs to call `reset()`.

        - truncated (bool): Whether the truncation condition outside the scope of the
        MDP is satisfied. Typically, this is a timelimit, but could also be used to
        indicate an agent physically going out of bounds. Can be used to end the episode
        prematurely before a terminal state is reached. If true, the user needs to call
        `reset()`.

        - info (dict): Contains auxiliary diagnostic information (helpful for debugging,
        learning, and logging). This might, for instance, contain: metrics that describe
        the agent’s performance state, variables that are hidden from observations, or
        individual reward terms that are combined to produce the total reward. In OpenAI
        Gym <v26, it contains `TimeLimit.truncated` to distinguish truncation and
        termination, however this is deprecated in favour of returning terminated and
        truncated variables.
        """

        # set the action on the controller
        self.controllers[0][:] = action
        # pass the action to the emulator as an unsigned byte
        simple_nes.step()

        observation = self.observation
        reward = NESEnv._clip(self._reward, self.reward_range[0], self.reward_range[1])
        terminated = self._terminated
        truncated = self._truncated
        info = self._info

        self._did_step()

        return observation, reward, terminated, truncated, info

    @abstractmethod
    def _did_step(self) -> None:
        pass

    @abstractmethod
    def _will_reset(self) -> None:
        pass

    @abstractmethod
    def _did_reset(self) -> None:
        pass

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Resets the environment to an initial internal state, returning an initial
        observation and info.

        This method generates a new starting state often with some randomness to ensure
        that the agent explores the state space and learns a generalised policy about
        the environment. This randomness can be controlled with the seed parameter
        otherwise if the environment already has a random number generator and `reset()`
        is called with `seed=None`, the RNG is not reset.

        Therefore, `reset()` should (in the typical use case) be called with a seed
        right after initialization and then never again.

        For Custom environments, the first line of `reset()` should be
        `super().reset(seed=seed)` which implements the seeding correctly.

        Parameters:
        - seed (optional int): The seed that is used to initialize the environment’s
        PRNG (`np_random`). If the environment does not already have a PRNG and
        `seed=None` (the default option) is passed, a seed will be chosen from some
        source of entropy (e.g. timestamp or /dev/urandom). However, if the environment
        already has a PRNG and `seed=None` is passed, the PRNG will not be reset. If you
        pass an integer, the PRNG will be reset even if it already exists. Usually, you
        want to pass an integer right after the environment has been initialized and
        then never again.

        - options (optional dict): Additional information to specify how the environment
        is reset (optional, depending on the specific environment).

        Returns:
        - observation (ObsType): Observation of the initial state. This will be an
        element of `observation_space` (typically a numpy array) and is analogous to the
        observation returned by `step()`.

        - info (dictionary): This dictionary contains auxiliary information
        complementing observation. It should be analogous to the info returned by
        `step()`.
        """

        super().reset(seed=seed)

        self._will_reset()

        if simple_nes.has_backup:
            simple_nes.restore()
        else:
            simple_nes.reset()

        self._did_reset()

        observation = self.observation
        info = self._info

        return observation, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """
        Compute the render frames as specified by render_mode during the initialization
        of the environment.

        The environment’s metadata render modes (`env.metadata[“render_modes”]`) should
        contain the possible ways to implement the render modes. In addition, list
        versions for most render modes is achieved through `gymnasium.make` which
        automatically applies a wrapper to collect rendered frames.

        Note: As the render_mode is known during `__init__`, the objects used to render
        the environment state should be initialised in `__init__`.

        By convention, if the render_mode is:
        - None (default): no render is computed.

        - "human": The environment is continuously rendered in the current display or
        terminal, usually for human consumption. This rendering should occur during
        `step()` and `render()` doesn’t need to be called. Returns None.

        - "rgb_array": Return a single frame representing the current state of the
        environment. A frame is a `np.ndarray` with shape (x, y, 3) representing RGB
        values for an x-by-y pixel image.

        - "ansi": Return a string (`str`) or `StringIO.StringIO` containing a
        terminal-style text representation for each time step. The text can include
        newlines and ANSI escape sequences (e.g. for colors).

        - "rgb_array_list" and "ansi_list": List based version of render modes are
        possible (except Human) through the wrapper,
        `gymnasium.wrappers.RenderCollection` that is automatically applied during
        `gymnasium.make(..., render_mode="rgb_array_list")`. The frames collected are
        popped after `render()` is called or `reset()`.
        """

        if self.render_mode == "rgb_array":
            return self.screen

    def close(self) -> None:
        """
        After the user has finished using the environment, close contains the code
        necessary to "clean up" the environment.

        This is critical for closing rendering windows, database or HTTP connections.
        """

        # purge the environment from C++ memory
        simple_nes.close()
        # if there is an image viewer open, delete it
        if self.viewer is not None:
            self.viewer.close()

    def _skip_frame(self, action: ActType) -> None:
        """
        Advance a frame in the emulator with an action.
        """

        # set the action on the controller
        self.controllers[0][:] = action
        # perform a step on the emulator
        simple_nes.step()

    @staticmethod
    def _backup() -> None:
        simple_nes.backup()

    @staticmethod
    def _restore() -> None:
        simple_nes.restore()

    @staticmethod
    def _clip(
        value: SupportsFloat, smallest: SupportsFloat, largest: SupportsFloat
    ) -> SupportsFloat:
        return max(smallest, min(value, largest))
