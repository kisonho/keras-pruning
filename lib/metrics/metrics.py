from ..core import tensorflow as tf
from ..core.typing import Any
from ..sparsity.prune import PruningMethod
from tensorflow.keras.metrics import Metric

class PruningRatio(Metric):
    _method: PruningMethod
    _pruning_ratio: float

    def __init__(self, method: PruningMethod, **kwargs):
        super().__init__(**kwargs)
        self._method = method

    def result(self) -> float:
        return self._pruning_ratio

    def update_state(self, *args: Any, **kwargs: Any) -> float:
        # initialize
        size: int = 0
        total_size: int = 0

        # loop for masks
        for m in self._method.masks:
            total_size += tf.size(m)
            size += tf.cast(tf.reduce_sum(m), tf.int32) # type: ignore

        # calculate pruning ratio
        self._pruning_ratio = 1 - size / total_size
        return self._pruning_ratio
