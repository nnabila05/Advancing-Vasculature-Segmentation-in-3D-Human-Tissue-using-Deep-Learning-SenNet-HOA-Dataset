# Advancing Vasculature Segmentation in 3D Human Tissue using Deep Learning: Insights from SenNet + HOA Dataset
## Data Source 
The SenNet + HOA - Hacking the Human Vasculature in 3D data set is collected from Kaggle. To download the data or know more  about the dataset and data description, please visit.
<https://www.kaggle.com/competitions/blood-vessel-segmentation>
## Description
In this study, we evaluated the performance of variations of  three distinct models—UNet, ResNet, and Vision Transformer (ViT)—on the task of binary vascular segmentation using the SenNet + HOA - Hacking the Human Vasculature in 3D dataset. Each architecture brings unique strengths to the table—ResNet's deep learning efficiency, U-Net's specialized design for medical imaging, and ViT's innovative use of Transformers. The objective of this study is to find the most suitable model for the blood vessel segmentation task. This project promises to have significant practical impacts, particularly in enhancing diagnostic processes and improving patient outcomes.
## Model Overview
This project utilizes variationa of three state-of-the-art models for binary vascular segmentation:

**ResNet (Residual Network)** - Employs deep residual learning for effective deep network training.

**U-Net** - Optimized for medical image segmentation, featuring a U-shaped architecture for precise localization.

**Vision Transformers (ViT)** - Adapts Transformer architecture for image analysis, focusing on global dependencies within the images.

The models were evaluated based on their performance in accurately classifying pixels within vascular structures, utilizing metrics such as the Dice coefficient and Intersection over Union (IoU) along with pixel wise accuracy, precision, recall and F1 scores.

## Results
Performance Metrics: 

All three models performed quite well in the binary vascular segmentation task in terms of accuracy and precision. In the case of dice coefficient and IoU UNet did significantly better. Overall UNet gave the best performance over all the metrics followed by ViT. However, the ViT model was also computationally more expensive and required more training hours to give similar metrics.

