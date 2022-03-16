from ..core.typing import Enum, Optional, Union
from ..core import Model
from ..sparsity.prune import PruningMethod
from .schedules import PruningScheduler
from tensorflow.keras.callbacks import * # type: ignore

class UpdateFrequency(Enum):
    BATCH = 1
    EPOCH = 0

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
    update_freq: Union[int, UpdateFrequency]

    def __init__(self, method: PruningMethod, schedule: Optional[PruningScheduler]=None, update_freq: Union[int, UpdateFrequency] = UpdateFrequency.EPOCH) -> None:
        """
        Constructor

        - Parameters:
            - method: A `PruningMethod` to update the mask
        """
        self.method = method
        self.schedule = schedule
        self.update_freq = update_freq

    def _update(self) -> None:
        # update pruning ratio
        if self.schedule is not None:
            new_pruning_ratio = self.schedule.update()
        else: new_pruning_ratio = None

        # update mask
        if self.model is not None and new_pruning_ratio is not None:
            self.method.pruning_ratio = new_pruning_ratio
            self.method.compute_mask(self.model)

    def on_batch_end(self, batch: int, **kwargs):
        if not isinstance(self.update_freq, UpdateFrequency):
            if batch % self.update_freq == 0: self._update()
        elif self.update_freq == UpdateFrequency.BATCH:
            self._update()

    def on_epoch_end(self, *args, **kwargs) -> None:
        # update pruning ratio
        if self.update_freq == UpdateFrequency.EPOCH:
            self._update()