from unet_1 import conv_block, encoder_block, decoder_block, build_unet
from vit_1  import mlp, Patches,PatchEncoder,create_vit_classifier
from resnet_1  import residual_block,build_resnet16
import os
import keras
import keras_cv
import tensorflow as tf
import tensorflow_io as tfio
import cv2
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import glob
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda,Add
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

print("OpenCV version:", cv2.__version__)
print("Tensorflow version:", tf.__version__)
print("Keras version:", keras.__version__)  
print("keras_cv version:", keras_cv.__version__)
print("Tensorflow_io version:", tfio.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)  

print("PIL version:", Image.__version__)



current_dir=os.getcwd()
print(f"Current directory:{current_dir}")

base_path = os.getcwd()

#Creating a dataframe of all tthe image and mask paths paths for easier handling
label_path = sorted(glob.glob(f"{base_path}/train/*/labels/*tif"))
print("Example paths:", label_path[:5])
df = pd.DataFrame({"mask_path":label_path})
df['dataset'] = df.mask_path.map(lambda x: x.split('/')[-3])
df['slice'] = df.mask_path.map(lambda x: x.split('/')[-1].replace(".tif", ""))
df = df[~df.dataset.str.contains("kidney_3_dense")]# kidney_3_dense doennt have labels and repeated again in kidney_3_sparse


df['image_path'] = df.mask_path.str.replace("label","image")
print(df.dataset.value_counts())
# kidney_3_sparse contais the images of the dense ones too.



#creating image and mask datsets and resizing at 256
SIZE = 256
image_dataset = []
mask_dataset = []


for image_path in df['image_path']:
    if image_path.endswith('.tif'):
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)#reading images in greyscale

        image = Image.fromarray(image) # Convert to PIL Image to use Image.resize
        image = image.resize((SIZE, SIZE))

        image_dataset.append(np.array(image))# converting to 
    else:
        print(f"Failed to load image: {image_path}")

for mask_path in df['mask_path']:
    if mask_path.endswith('.tif'):
        mask = cv2.imread(mask_path,0)

        mask = Image.fromarray(mask)  # Convert to PIL Image to use Image.resize
        mask = mask.resize((SIZE, SIZE))
        mask_dataset.append(np.array(mask))  # Append the processed mask to the dataset list
    else:
         print(f"Failed to load mask: {mask_path}")



#expanding the dimentions and normalizing, preparing for the model input
image_dataset = np.array(image_dataset)/255.
image_dataset = np.expand_dims((np.array(image_dataset)),3)
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.
mask_dataset[mask_dataset > 0.5] = 1  # thresholding the greyscale values to binary for better prediction
mask_dataset[mask_dataset <= 0.5] = 0
print("Labels in the mask are : ", np.unique(mask_dataset))
print("Image data shape is: ", image_dataset.shape)
print("Mask data shape is: ", mask_dataset.shape)
print("Max pixel value in image is: ", image_dataset.max())
print("Labels in the mask are : ", np.unique(mask_dataset))


"""creating test and traing datsets. 
Test datset is created by splitting from the traing data as the test data provided for the competion did not have labels. 
Due to the scope of this project and metrics requirments a part of training data was seperated as test data"""

X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)
#checking shape of X_train ,y_train, X_test and y_test.
print("X_train shape",X_train.shape)
print("X_test shape",X_test.shape)
print("y_train shape",y_train.shape)
print("y_test shape",y_test.shape)

IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

#dice score and IoU calculation
def compute_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Function to calculate Dice Score
def compute_dice_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    dice_score = 2 * np.sum(intersection) / (np.sum(y_true) + np.sum(y_pred))
    return dice_score



##UNet
model = build_unet(input_shape, n_classes=1)
model.compile(optimizer=Adam(learning_rate = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
#training till 4 epochs as it gave the most accuracy and least loss
history = model.fit(X_train, y_train,
                    batch_size = 16,
                    verbose=1,
                    epochs=4,
                    validation_split=0.2,
                    shuffle=False)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test loss_Unet:", test_loss)
print("Test accuracy_Unet:", test_accuracy)

#calculating metrics
unet_predictions = model.predict(X_test)
unet_y_pred_thresholded = unet_predictions > 0.5
unet_y_pred_thresholded=unet_y_pred_thresholded.astype(int)

y_test_flat = y_test.flatten()
unet_pred_flat = unet_y_pred_thresholded.flatten()

precision = precision_score(y_test_flat, unet_pred_flat)
recall = recall_score(y_test_flat, unet_pred_flat)
f1 = f1_score(y_test_flat, unet_pred_flat)
print(" Unet Precision:", precision)
print(" Unet Recall:", recall)
print(" Unet F1 Score:", f1)

accuracy = accuracy_score(y_test_flat, unet_pred_flat)
print("Unet Pixel-wise Accuracy:", accuracy)

iou = compute_iou(y_test_flat, unet_pred_flat)
dice = compute_dice_score(y_test_flat, unet_pred_flat)

print("Unet IoU:", iou)
print("Unet Dice Score:", dice)


class_report = classification_report(y_test_flat, unet_pred_flat)
print("Unet Classification Report:\n", class_report)





##resnet##
res_model = build_resnet16(input_shape=input_shape)
res_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(res_model.summary())

res_history = res_model.fit(X_train, y_train,
                    batch_size = 16,
                    verbose=1,
                    epochs=4,
                    validation_split=0.2,
                    shuffle=False)




res_test_loss, res_test_accuracy = res_model.evaluate(X_test, y_test)
print("ResNet Test loss:", res_test_loss)
print("ResNet Test accuracy:", res_test_accuracy)

res_predictions = res_model.predict(X_test)
res_y_pred_thresholded = res_predictions > 0.5
res_y_pred_thresholded=res_y_pred_thresholded.astype(int)
res_y_pred_flat = res_y_pred_thresholded.flatten()

res_precision = precision_score(y_test_flat, res_y_pred_flat)
res_recall = recall_score(y_test_flat, res_y_pred_flat)
res_f1 = f1_score(y_test_flat, res_y_pred_flat)
print("ResNet Precision:", res_precision)
print("ResNet Recall:", res_recall)
print("ResNet F1 Score:", res_f1)

res_accuracy = accuracy_score(y_test_flat, res_y_pred_flat)
print("ResNet Pixel-wise Accuracy:", res_accuracy)

res_iou = compute_iou(y_test_flat, res_y_pred_flat)
res_dice = compute_dice_score(y_test_flat, res_y_pred_flat)

print("ResNet IoU:", res_iou)
print("ResNet Dice Score:", res_dice)


res_class_report = classification_report(y_test_flat, res_y_pred_flat)
print("ResNet Classification Report:\n", res_class_report)


##ViT Model 
#parameters
patch_size = 16  # Patch size that evenly divides both dimensions of the image
num_patches = (input_shape[0] // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_layers = 8
transformer_units = [projection_dim * 2, projection_dim]
mlp_head_units = [2048, 1024]
num_classes = 1  # For binary classification

vit_model = create_vit_classifier()
print(vit_model.summary())

# Compile the model
vit_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

vit_history = vit_model.fit(X_train, y_train,
                    batch_size = 6,
                    verbose=1,
                    epochs=50,
                    validation_split=0.2,
                    shuffle=False)


vit_test_loss, vit_test_accuracy = vit_model.evaluate(X_test, y_test)
vit_predictions = vit_model.predict(X_test)
vit_y_pred_thresholded = vit_predictions > 0.5
vit_y_pred_thresholded=vit_y_pred_thresholded.astype(int)

vit_y_pred_flat = vit_y_pred_thresholded.flatten()

vit_precision = precision_score(y_test_flat, vit_y_pred_flat)
vit_recall = recall_score(y_test_flat, vit_y_pred_flat)
vit_f1 = f1_score(y_test_flat, vit_y_pred_flat)
print("ViT Precision:", vit_precision)
print("ViT Recall:", vit_recall)
print("ViTF1 Score:", vit_f1)

vit_accuracy = accuracy_score(y_test_flat, vit_y_pred_flat)
print("ViT Pixel-wise Accuracy:", vit_accuracy)

vit_iou = compute_iou(y_test_flat, vit_y_pred_flat)
vit_dice = compute_dice_score(y_test_flat, vit_y_pred_flat)

print("ViT IoU:", vit_iou)
print(" ViT Dice Score:", vit_dice)
vit_class_report = classification_report(y_test_flat, vit_y_pred_flat)
print("ViT Classification Report:\n", vit_class_report)

metrics_data = {
    "Model": ["U-Net", "ResNet", "Vision Transformer"],
    "Accuracy": [accuracy, res_accuracy, vit_accuracy],
    "Precision": [precision, res_precision, vit_precision],
    "Recall": [recall, res_recall, vit_recall],
    "F1 Score": [f1, res_f1, vit_f1],
    "Dice Score": [dice, res_dice, vit_dice],
    "IoU": [iou, res_iou, vit_iou]
}

# Create a DataFrame
metrics_df = pd.DataFrame(metrics_data)

# Display the DataFrame
print(metrics_df)