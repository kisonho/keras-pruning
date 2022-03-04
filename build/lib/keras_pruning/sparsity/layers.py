# import typing modules
from __future__ import annotations
from typing import Any, Callable, Optional, Protocol, Type, runtime_checkable

# import required modules
import abc, tensorflow as tf
from tensorflow.keras import layers

@runtime_checkable
class _PrunableLayer(Protocol):
    """
    A prunable layer protocol
    
    - Properties:
        - kernel: Either A `tf.Tensor` of a `tf.Variable` of the kernel
        - name: A `str` of current layer name
        - trainable_variables: A `list` of trainable `tf.Variable`
    """
    @abc.abstractproperty
    def bias(self) ->Optional[tf.Variable]:
        raise NotImplementedError

    @bias.setter
    def bias(self, bias: tf.Tensor) -> None:
        raise NotImplementedError

    @abc.abstractproperty
    def built(self) -> bool:
        raise NotImplementedError
    
    @bias.setter
    @abc.abstractmethod
    def built(self, build: bool) -> None:
        raise NotImplementedError

    @abc.abstractproperty
    def kernel(self) -> tf.Variable:
        raise NotImplementedError

    @kernel.setter
    @abc.abstractmethod
    def kernel(self, kernel: tf.Tensor) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> tf.Tensor:
        return NotImplementedError

    @abc.abstractmethod
    def build(self, input_shape: Any) -> None:
        return NotImplementedError

class PrunedLayer(layers.Layer):
    """
    A layer that is pruned with pruning wrap
    
    - Properties:
        - masks: A `list` of masks in `tf.Variable`
        - orig_vars: A `list` of original variables in `tf.Variable`
        - prunable_variables: A `list` of `tf.Variable` which is prunable
    """
    # properties
    _target_layer: _PrunableLayer
    mask_hook: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
    mask: tf.Variable
    orig_var: tf.Variable

    @property
    def kernel(self) -> tf.Variable:
        return self.orig_var

    @kernel.setter
    def kernel(self, kernel: tf.Tensor) -> None:
        self._target_layer.kernel = kernel

    @property
    def target(self) -> layers.Layer:
        assert isinstance(self._target_layer, layers.Layer), "[Pruning Error]: Target layer is not a keras Layer."
        return self._target_layer

    def __init__(self, target: _PrunableLayer, mask_hook: Callable[[tf.Tensor, tf.Tensor], tf.Tensor], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._target_layer = target
        self.mask_hook = mask_hook

    def build(self, input_shape: Any) -> None:
        self._target_layer.build(input_shape)
        assert isinstance(self._target_layer.kernel, tf.Variable), "[Pruning Error]: kernel is not a valid tf.Variable."
        self.orig_var = self._target_layer.kernel
        self.mask = tf.Variable(tf.ones(self.orig_var.shape, name=f"{self.orig_var.name}_mask"), trainable=False)

    def call(self, *args, **kwargs) -> tf.Tensor:
        self.kernel = self.mask_hook(self.orig_var, self.mask)
        return self._target_layer(*args, **kwargs)

    def prune(self, threshold: tf.Tensor) -> None:
        """
        Prune the current layer
        
        - Parameters
            - pruning_ratio: A `float` of pruning ratio
        """
        mask = tf.greater(self.orig_var, threshold)
        mask = tf.cast(mask, tf.float32)
        self.mask.assign(mask)

    def get_config(self) -> dict[str, Any]:
        # get target type
        type_name: str = self.target.__class__.__name__

        cfg: dict[str, Any] = {
            "name": self.name,
            "target": self.target.get_config(),
            "type": type_name
        }
        return cfg

    @classmethod
    def from_config(cls: Type[PrunedLayer], cfg: dict[str, Any]) -> PrunedLayer:
        # build target
        target_cfg = cfg["target"]
        target_type: Type[layers.Layer] = getattr(layers, cfg["type"])
        target = target_type.from_config(target_cfg)

        # build pruned layer
        return cls(target, lambda orig_var, mask: orig_var * mask, name=cfg["name"]) # type: ignore
