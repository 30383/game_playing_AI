import ctypes
import glob
import os
import sys

import numpy as np


class SimpleNES:
    def __init__(self):
        self.lib = self._load_lib()

        self._env = None
        self.has_backup = False

    @staticmethod
    def _load_lib():
        # the path to the directory this file is in
        module_path = os.path.dirname(__file__)
        # the pattern to find the C++ shared object self.library
        so_path = "lib_nes_env*"
        # the absolute path to the C++ shared object self.library
        lib_path = os.path.join(module_path, so_path)
        # load the self.library from the shared object file
        try:
            print(lib_path)
            so = ctypes.CDLL(glob.glob(lib_path)[0])
        except IndexError:
            raise OSError("missing static self.lib_nes_env*.so self.library!")

        accessible_functions = {
            "Width": {"argtypes": None, "restype": ctypes.c_uint},
            "Height": {"argtypes": None, "restype": ctypes.c_uint},
            "Initialize": {"argtypes": [ctypes.c_wchar_p], "restype": ctypes.c_void_p},
            "Controller": {
                "argtypes": [ctypes.c_void_p, ctypes.c_uint],
                "restype": ctypes.c_void_p,
            },
            "Screen": {"argtypes": [ctypes.c_void_p], "restype": ctypes.c_void_p},
            "Memory": {"argtypes": [ctypes.c_void_p], "restype": ctypes.c_void_p},
            "Reset": {"argtypes": [ctypes.c_void_p], "restype": None},
            "Step": {"argtypes": [ctypes.c_void_p], "restype": None},
            "Backup": {"argtypes": [ctypes.c_void_p], "restype": None},
            "Restore": {"argtypes": [ctypes.c_void_p], "restype": None},
            "Close": {"argtypes": [ctypes.c_void_p], "restype": None},
        }

        lib = {}

        for key, values in accessible_functions.items():
            lib[key.lower()] = getattr(so, key)
            lib[key.lower()].argtypes = values["argtypes"]
            lib[key.lower()].restype = values["restype"]

        return lib

    @property
    def width(self) -> int:
        return self.lib["width"]()

    @property
    def height(self) -> int:
        return self.lib["height"]()

    @property
    def env(self) -> ctypes.c_void_p:
        return self._env

    @env.setter
    def env(self, path: str = None):
        self._env = None if path is None else self.lib["initialize"](path)

    def controller(self, port: int) -> np.ndarray:
        # get the address of the controller
        address = self.lib["controller"](self.env, port)
        # create a memory buffer using the ctypes pointer for this vector
        buffer = ctypes.cast(address, ctypes.POINTER(ctypes.c_byte * 1)).contents
        # create a NumPy buffer from the binary data and return it
        return np.frombuffer(buffer, dtype="uint8")

    def screen(self) -> np.ndarray:
        # get the address of the screen
        address = self.lib["screen"](self.env)
        # create a buffer from the contents of the address location
        buffer = ctypes.cast(address, ctypes.POINTER(self.screen_tensor)).contents
        # create a NumPy array from the buffer
        screen = np.frombuffer(buffer, dtype="uint8")
        # reshape the screen from a column vector to a tensor
        screen = screen.reshape(self.screen_shape_32_bit)
        # flip the bytes if the machine is little-endian (which it likely is)
        if sys.byteorder == "little":
            # invert the little-endian BGRx channels to big-endian xRGB
            screen = screen[:, :, ::-1]
        # remove the 0th axis (padding from storing colors in 32 bit)
        return screen[:, :, 1:]

    def memory(self) -> np.ndarray:
        # get the address of the RAM
        address = self.lib["memory"](self.env)
        # create a buffer from the contents of the address location
        buffer_ = ctypes.cast(address, ctypes.POINTER(ctypes.c_byte * 0x800)).contents
        # create a NumPy array from the buffer
        return np.frombuffer(buffer_, dtype="uint8")

    def reset(self):
        self.lib["reset"](self.env)

    def step(self):
        self.lib["step"](self.env)

    def backup(self):
        self.has_backup = True
        self.lib["backup"](self.env)

    def restore(self):
        self.lib["restore"](self.env)

    def close(self):
        self.lib["close"](self.env)

    @property
    def screen_shape_24_bit(self):
        # shape of the screen as 24-bit RGB (standard for NumPy)
        return self.height, self.width, 3

    @property
    def screen_shape_32_bit(self):
        # shape of the screen as 32-bit RGB (C++ memory arrangement)
        return self.height, self.width, 4

    @property
    def screen_tensor(self):
        # create a type for the screen tensor matrix from C++
        return ctypes.c_byte * int(np.prod(self.screen_shape_32_bit))


simple_nes = SimpleNES()
