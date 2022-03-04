# import typing modules
from typing import Callable, Optional

# import required modules
import tensorflow as tf
from tensorflow.keras import Input, Model, layers, regularizers

def ResNet56(input_shape: tuple[int, int, int]=(32, 32, 3), num_classes: int=10, weight_decay: float=1e-4) -> Model:
    '''
    Build ResNet 56 model

    - Parameters:
        - input_shape: A `tuple` of the 3-dimension input size in `int`
        - num_classes: An `int` of the prediction classes
        - weight_decay: A `float` of weight decay
    '''
    # build resnet layer function
    def resnet_layer(inputs: tf.Tensor, num_filters: int=16, kernel_size: int=3, strides: int=1, activation: Optional[Callable[[tf.Tensor], tf.Tensor]]=tf.nn.relu, use_batch_normalization: bool=True, is_conv_first: bool=True) -> tf.Tensor:
        # define conv layer
        conv_layer = layers.Conv2D(num_filters, kernel_size=kernel_size, kernel_regularizer=regularizers.l2(weight_decay), strides=strides, padding='same')

        x = inputs
        if is_conv_first is True:
            x = conv_layer(x)
            if use_batch_normalization is True:
                x = layers.BatchNormalization()(x)
            if activation is not None:
                x = activation(x)
        else:
            if use_batch_normalization is True:
                x = layers.BatchNormalization()(x)
            if activation is not None:
                x = activation(x)
            x = conv_layer(x)
        return x
        
    # Start model definition.
    num_filters_in: int = 16
    num_res_blocks: int = 6
    num_filters_out: int = 0

    # initialize inputs
    inputs: tf.Tensor = Input(shape=input_shape)

    # v2 performs layers.Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, is_conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = tf.nn.relu
            use_batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    use_batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x, num_filters=num_filters_in, kernel_size=1, strides=strides, activation=activation, use_batch_normalization=use_batch_normalization, is_conv_first=False)
            y = resnet_layer(inputs=y, num_filters=num_filters_in, is_conv_first=False)
            y = resnet_layer(inputs=y, num_filters=num_filters_out, kernel_size=1, is_conv_first=False)
            
            # first block add
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x, num_filters=num_filters_out, kernel_size=1, strides=strides, activation=None, use_batch_normalization=False)
                
            # add layer
            x = x + y

        # update input filters
        num_filters_in = num_filters_out 

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = layers.AveragePooling2D(pool_size=7)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(num_classes)(x)
    y: tf.Tensor = tf.nn.softmax(x)

    # wrap to model
    return Model(inputs=inputs, outputs=y, name='resnet56')