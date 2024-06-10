import tensorflow as tf
import keras.layers as layers
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
import keras


#parameters
input_shape = (256, 256, 1)  # Grayscale image shape
patch_size = 16  # Patch size that evenly divides both dimensions of the image
num_patches = (input_shape[0] // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_layers = 8
transformer_units = [projection_dim * 2, projection_dim]
mlp_head_units = [2048, 1024]
num_classes = 1 


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation='gelu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Patch extraction layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        # Extract patches
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID')
        # Flatten patches
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        x = self.projection(patches)
        return x + self.position_embedding(positions)

# Building the ViT model
def create_vit_classifier():
    inputs = keras.Input(shape=input_shape)
    # Create patches
    patches = Patches(patch_size)(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Transformer blocks
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    # Final processing
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, mlp_head_units, dropout_rate=0.5)
    x = layers.Reshape((input_shape[0] // patch_size, input_shape[1] // patch_size, projection_dim))(representation)
    x = layers.Conv2DTranspose(filters=projection_dim, kernel_size=3, strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(filters=projection_dim // 2, kernel_size=3, strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')(x)
    outputs = layers.Conv2D(1, kernel_size=1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="ViT")
    return model