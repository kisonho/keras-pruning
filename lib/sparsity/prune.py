# import typing modules
from __future__ import annotations
from typing import Type, Union

# import required modules
import abc, tensorflow as tf
from tensorflow.keras import layers, models, Model

# import core modules
from .layers import _PrunableLayer, PrunedLayer

class PruningMethod(abc.ABC):
    """
    A basic pruning method

    - Properties:
        - masks: A `list` of the mask `tf.Variable`
        - pruning_ratio: A `float` of pruning ratio
        - skipped_layers: A `list` of the name of skipped layers in `str`
    """
    # parameters
    __pruned_layer_type: Type[PrunedLayer]
    __pruning_ratio: float
    __skipped_layers: list[str]
    masks: list[tf.Variable]

    @property
    def pruning_ratio(self) -> float:
        return self.__pruning_ratio

    @pruning_ratio.setter
    def pruning_ratio(self, p: float) -> None:
        assert p > 0 and p < 1, f"[Pruning Error]: Pruning ratio must between (0,1), got {p}"
        self.__pruning_ratio = p

    @property
    def skipped_layers(self) -> list[str]:
        return self.__skipped_layers

    def __init__(self, pruning_ratio: float, skipped_layers: list[str]=[], pruned_layer_type: Type[PrunedLayer] = PrunedLayer) -> None:
        """
        Constructor
        
        - Parameters:
            - pruning_ratio: A `float` of pruning ratio
        """
        self.__skipped_layers = skipped_layers
        self.__pruned_layer_type = pruned_layer_type
        self.masks = []
        self.pruning_ratio = pruning_ratio

    def _apply_pruning_wrap(self, layer: layers.Layer) -> layers.Layer:
        """
        Convert a `layers.Layer` into a `PrunedLayer`, a `layers.Layer` that is not prunable will not be converted

        - Parameters:
            - layer: A `layers.Layer` to be converted
        - Returns: Either a converted `PrunedLayer` or the original layer in `layers.Layer`
        """
        # check prunable
        if isinstance(layer, _PrunableLayer) and layer.name not in self.__skipped_layers:
            layer_name = f"{layer.name}_pruned"
            pruned_layer = self.__pruned_layer_type(layer, self.apply_mask, layer_name)
        else: return layer
        return pruned_layer

    def apply(self, model: Model) -> Model:
        """
        Apply pruning method to target model

        - Parameters:
            model: The target `Model`
        - Returns: A pruned `Model`
        """
        # compute mask
        pruned_model: Model = models.clone_model(model, clone_function=self._apply_pruning_wrap)
        self.masks = self.compute_mask(pruned_model)
        return pruned_model

    @staticmethod
    def apply_mask(var: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        Applies masks to target model

        - Parameters:
            var: A target `tf.Tensor`
            masks: A mask in `tf.Tensor` to be applied
        - Returns: A `tf.Tensor` of applied variable
        """
        return var * mask # type: ignore

    @abc.abstractmethod
    def compute_mask(self, model: Model) -> list[tf.Variable]:
        """
        Method to update the mask
        
        - Parameters:
            model: The target `Model`
        - Returns: A `list` of mask in `tf.Variable`
        """
        raise NotImplementedError

    @staticmethod
    def remove(layer: Union[layers.Layer, PrunedLayer]) -> layers.Layer:
        """
        Convert a `PrunableLayer` back to a `layers.Layer`, a traditional `layers.Layer` without pruning wrap will not be converted

        - Parameters:
            - layer: A `layers.Layer` to be converted
        - Returns: A `layers.Layer` with pruning wrap removed
        """
        # check prunable
        if isinstance(layer, PrunedLayer):
            return layer.target
        else: return layer

class GlobalL1Unstructured(PruningMethod):
    """Global L1 unstructured pruning method"""
    # parameters
    _prunable_layers: list[PrunedLayer] = []

    def compute_mask(self, model: Model) -> list[tf.Variable]:
        # calculate global l1 threshold
        self._prunable_layers = [l for l in model.layers if isinstance(l, PrunedLayer)]
        flattened_vars: list[tf.Tensor] = [tf.reshape(l.orig_var, (-1)) for l in self._prunable_layers]
        vars: tf.Tensor = tf.concat(flattened_vars, axis=0)
        vars = tf.sort(vars)
        threshold_index = min(int(vars.shape[0] * self.pruning_ratio), vars.shape[0] - 1) # type: ignore
        threshold: tf.Tensor = vars[threshold_index] # type: ignore

        # prune each layer
        for layer in self._prunable_layers:
            layer.prune(threshold)
        return [l.mask for l in self._prunable_layers]

class GlobalL1STEUnstructured(PruningMethod):
    """Global L1 unstructured pruning method with STE"""
    @staticmethod
    @tf.custom_gradient
    def apply_mask(var: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        return var * mask, lambda dy: (dy, tf.zeros_like(mask)) # type: ignore

def remove(pruned_model: Model, pruning_method: Type[PruningMethod]=PruningMethod) -> Model:
    """
    Removes all `PrunedLayer` pruning wrap inside a `Model`, other layers will not be effected

    - Parameters:
        - pruned_model: A `Model` that is pruned
    - Returns: A non-pruned `Model`
    """
    # clone existing model
    model: Model = models.clone_model(pruned_model, clone_function=pruning_method.remove)
    return model
