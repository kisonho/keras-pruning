# import typing modules
from .. import core
from ..core.typing import Any, Optional, Type
from ..core import layers
from tensorflow.keras.models import * # type: ignore

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
    loaded_model = core.load_model(*args, custom_objects=objects, **kwargs)
    return loaded_model