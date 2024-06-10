import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2

def residual_block(x, filters, kernel_size=3, dropout_rate=0.5, l1_reg=1e-5, l2_reg=1e-4):
    """A ResNet-style residual block."""
    y = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(dropout_rate)(y)

    y = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l1_l2(l1_reg, l2_reg))(y)
    y = BatchNormalization()(y)

    x = Add()([x, y])  # Skip connection
    return Activation('relu')(x)

def build_resnet16(input_shape, num_filters=64, num_blocks=5, dropout_rate=0.5, l1=1e-5, l2=1e-4):
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters, (7, 7), strides=2, padding='same', kernel_regularizer=l1_l2(l1, l2))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Adding multiple residual blocks
    for _ in range(num_blocks):
        x = residual_block(x, num_filters, dropout_rate=dropout_rate, l1_reg=l1, l2_reg=l2)

    # Upsampling and final convolution
    x = tf.keras.layers.Conv2DTranspose(num_filters, (3, 3), strides=2, padding='same')(x)
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)  # Output layer for binary segmentation

    model = Model(inputs=inputs, outputs=x, name= "ResNet")
    return model