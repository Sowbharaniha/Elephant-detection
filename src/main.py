import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision
set_global_policy('mixed_float16')

# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import zipfile

# Unzip the dataset (relative path)
zip_path = './data/EleData.zip'
extract_path = './data/dataset'
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Set dataset paths and hyperparameters
train_dir = os.path.join(extract_path, 'train')
valid_dir = os.path.join(extract_path, 'valid')
train_annotations = os.path.join(train_dir, '_annotations.csv')
valid_annotations = os.path.join(valid_dir, '_annotations.csv')
img_height, img_width = 256, 256
batch_size = 16
epochs = 5

# Load annotations
train_df = pd.read_csv(train_annotations)
valid_df = pd.read_csv(valid_annotations)

# Map class to 1 for Elephant
train_df['class'] = 1
valid_df['class'] = 1

# Get all images
all_train_images = set(os.listdir(train_dir))
all_valid_images = set(os.listdir(valid_dir))

# Get annotated image sets
annotated_train_images = set(train_df['filename'])
annotated_valid_images = set(valid_df['filename'])

# Get non-elephant image lists
non_elephant_train_images = list(all_train_images - annotated_train_images)
non_elephant_valid_images = list(all_valid_images - annotated_valid_images)

# Dummy DataFrames for class=0
not_elephant_train_df = pd.DataFrame({'filename': non_elephant_train_images, 'class': 0})
not_elephant_valid_df = pd.DataFrame({'filename': non_elephant_valid_images, 'class': 0})

# Merge with the original DataFrames
train_df = pd.concat([train_df[['filename', 'class']], not_elephant_train_df], ignore_index=True)
valid_df = pd.concat([valid_df[['filename', 'class']], not_elephant_valid_df], ignore_index=True)

# Convert classes to string for ImageDataGenerator
train_df['class'] = train_df['class'].astype(str)
valid_df['class'] = valid_df['class'].astype(str)

# Print class distributions
print("\nTrain class distribution:")
print(train_df['class'].value_counts())
print("\nValid class distribution:")
print(valid_df['class'].value_counts())

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=train_dir,
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)
valid_generator = valid_datagen.flow_from_dataframe(
    valid_df,
    directory=valid_dir,
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Build the model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=50,
    validation_data=valid_generator
)

# Save the trained model
os.makedirs('./models', exist_ok=True)
model.save('./models/elephant_detector_1.h5')
print("✅ Model saved to ./models/elephant_detector_1.h5")

# Plot metrics
def plot_metrics(history):
    plt.figure(figsize=(12, 4))
    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Precision
    plt.subplot(1, 3, 2)
    plt.plot(history.history['precision'], label='Train Precision')
    plt.plot(history.history['val_precision'], label='Val Precision')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    # Recall
    plt.subplot(1, 3, 3)
    plt.plot(history.history['recall'], label='Train Recall')
    plt.plot(history.history['val_recall'], label='Val Recall')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Show the plots
plot_metrics(history)
