# import typing modules
from typing import Optional

# import required modules
from tensorflow.keras import Model
from tensorflow.keras.callbacks import * # type: ignore

# import call modules
from ..sparsity.prune import PruningMethod
from .schedules import PruningScheduler

class UpdatePruningMask(Callback):
    """
    The callback that updates pruning mask at each end of epoch

    - Parameters:
        - method: A `PruningMethod` to update the mask
        - model: An optional `Model` to be updated
        - scheduler: An optional `PruningScheduler` to update the pruning ratio
    """
    # parameters
    method: PruningMethod
    model: Optional[Model] = None
    schedule: Optional[PruningScheduler] = None

    def __init__(self, method: PruningMethod, schedule: Optional[PruningScheduler]=None) -> None:
        """
        Constructor

        - Parameters:
            - method: A `PruningMethod` to update the mask
        """
        self.method = method
        self.schedule = schedule

    def on_epoch_end(self, *args, **kwargs) -> None:
        # update pruning ratio
        if self.schedule is not None:
            new_pruning_ratio = self.schedule.update()
        else: new_pruning_ratio = None

        # update mask
        if self.model is not None and new_pruning_ratio is not None:
            self.method.pruning_ratio = new_pruning_ratio
            self.method.compute_mask(self.model)