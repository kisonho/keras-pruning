# import typing modules
from __future__ import annotations
from typing import Optional

# import required modules
import abc

class PruningScheduler:
    """
    The scheduler that schedules pruning ratio

    - Properties:
        initial_pruning_ratio: A `float` of the initial pruning ratio
        step: An `int` index of current step
    """
    # properties
    __initial_pruning_ratio: float
    __step: int = 0

    @property
    def initial_pruning_ratio(self) -> float:
        return self.__initial_pruning_ratio

    @property
    def step(self) -> int:
        return self.__step

    def __init__(self, initial_pruning_ratio: float) -> None:
        """
        Constructor

        - Parameters:
            - initial_pruning_ratio: The starting pruning ratio in `float`
        """
        self.__initial_pruning_ratio = initial_pruning_ratio

    def __call__(self) -> Optional[float]:
        """
        Direct calling function to get the current pruning ratio, a `None` value will result not updating pruning mask

        - Returns: An optional `float` of current pruning ratio
        """
        new_pruning_ratio = self.update()
        self.__step += 1
        return new_pruning_ratio

    @abc.abstractmethod
    def update(self) -> Optional[float]:
        """
        The method to update pruning ratio, a `None` value will result not updating pruning mask

        - Returns: An optional `float` of current pruning ratio
        """
        raise NotImplementedError

class ConstantPruningScheduler(PruningScheduler):
    """
    The scheduler that gives the same pruning ratio for each step

    - Properties:
        initial_step: An `int` index of update step
    """
    # properties
    __initial_step: int = 0

    @property
    def initial_step(self) -> int:
        return self.__initial_step

    def __init__(self, initial_pruning_ratio: float, initial_step: int=0) -> None:
        """
        Constructor

        - Parameters:
            - initial_pruning_ratio: The starting pruning ratio in `float`
            - initial_step: The first step index in `int` to update pruning mask
        """
        super().__init__(initial_pruning_ratio)
        self.__initial_step = initial_step

    def update(self) -> Optional[float]:
        return self.initial_pruning_ratio if self.step >= self.initial_step else None
