import numpy as np


class Randomizer:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(42)

    def uniform(self, min: float, max: float) -> float:
        return self.rng.uniform(min, max, None)
