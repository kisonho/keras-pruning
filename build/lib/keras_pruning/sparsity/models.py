# import typing modules
from typing import Any, Optional, Type

# import required modules
from tensorflow.keras import layers
from tensorflow.keras.models import * # type: ignore
from tensorflow.keras.models import load_model as _load_model

# import core modules
from .layers import PrunedLayer

def load_model(*args, custom_objects: Optional[dict[str, Type[layers.Layer]]]=None, **kwargs: Any) -> Optional[Any]:
    # define objects
    objects: Optional[dict[str, Any]] = {
        "PrunedLayer": PrunedLayer
    }

    # update to custom objects
    if custom_objects is not None:
        objects.update(custom_objects)

    # load with pruned layer
    loaded_model = _load_model(*args, custom_objects=objects, **kwargs)
    return loaded_model