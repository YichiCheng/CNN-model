import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Constants
IMAGE_DIR = "D:/logging_camera_down/image_data/"
LOG_FILE = "C:/logging_data/log_file_4.txt"  # Adjust the file index as needed
IMG_HEIGHT, IMG_WIDTH = 66, 200  # Resize dimensions
BATCH_SIZE = 32
EPOCHS = 50
AUGMENTATION_PROB = 0.5  # Probability of applying augmentations

# Load and preprocess data
def load_data():
    """
    Load the image file paths and corresponding steering angles from the log file.
    """
    data = []
    with open(LOG_FILE, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            timestamp, left_frame, right_frame, front_frame, steering_angle = parts
            data.append((front_frame, float(steering_angle)))
    
    df = pd.DataFrame(data, columns=["front_frame", "steering_angle"])
    return df

def preprocess_image(image_path):
    """
    Load and preprocess an image.
    """
    img = cv2.imread(os.path.join(IMAGE_DIR, image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # Resize to match input dimensions
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Augmentation functions
def augment_image(image):
    """
    Apply random augmentations to the image.
    """
    if np.random.rand() < AUGMENTATION_PROB:
        # Flip horizontally
        image = tf.image.flip_left_right(image)
    if np.random.rand() < AUGMENTATION_PROB:
        # Adjust brightness
        image = tf.image.adjust_brightness(image, delta=np.random.uniform(-0.2, 0.2))
    if np.random.rand() < AUGMENTATION_PROB:
        # Adjust saturation
        image = tf.image.adjust_saturation(image, saturation_factor=np.random.uniform(0.8, 1.2))
    return image

def preprocess_and_augment(image_path):
    """
    Preprocess and augment the image.
    """
    img = preprocess_image(image_path)
    img = augment_image(img)
    return img

# Data generator with augmentation
def data_generator(df, batch_size, augment=False):
    """
    Generate batches of data for training.
    """
    num_samples = len(df)
    while True:
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data
        for offset in range(0, num_samples, batch_size):
            batch_samples = df[offset:offset + batch_size]
            images = []
            angles = []
            for _, row in batch_samples.iterrows():
                if augment:
                    image = preprocess_and_augment(row["front_frame"])
                else:
                    image = preprocess_image(row["front_frame"])
                steering_angle = row["steering_angle"]
                images.append(image)
                angles.append(steering_angle)
            yield np.array(images), np.array(angles)

# Define the CNN model
def create_model():
    """
    Define and compile a CNN model.
    """
    model = Sequential([
        # Convolutional layers
        Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and fully connected layers
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.5),  # Dropout for regularization
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)  # Single output for steering angle
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
    return model

# Main function
def main():
    # Load and split data
    df = load_data()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create data generators with augmentation for training and no augmentation for validation
    train_gen = data_generator(train_df, BATCH_SIZE, augment=True)
    val_gen = data_generator(val_df, BATCH_SIZE, augment=False)
    
    # Build the model
    model = create_model()
    model.summary()
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1, min_delta=0.001)

    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_df) // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=len(val_df) // BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )
    
    # Save the model
    model.save("steering_model_augmented.h5")
    print("Model saved as steering_model_augmented.h5")

if __name__ == "__main__":
    main()
