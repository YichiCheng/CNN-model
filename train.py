import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Constants
IMAGE_DIR = "logging/image_data/"
LOG_FILE = "logging/logging_data/log_file_0.txt"  # Adjust the file index as needed
IMG_HEIGHT, IMG_WIDTH = 66, 200  # Resize dimensions
BATCH_SIZE = 32
EPOCHS = 20

def load_data():
    """
    Load the image file paths and corresponding steering angles from the log file.
    """
    # Read the log file
    data = []
    with open(LOG_FILE, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            timestamp, left_frame, right_frame, front_frame, steering_angle = parts
            data.append((front_frame, float(steering_angle)))
    
    # Convert to DataFrame for convenience
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

def data_generator(df, batch_size):
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
                image = preprocess_image(row["front_frame"])
                steering_angle = row["steering_angle"]
                images.append(image)
                angles.append(steering_angle)
            yield np.array(images), np.array(angles)

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
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)  # Single output for steering angle
    ])
    model.compile(optimizer=Adam(lr=0.0001), loss='mse', metrics=['mae'])
    return model

def main():
    # Load and split data
    df = load_data()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create data generators
    train_gen = data_generator(train_df, BATCH_SIZE)
    val_gen = data_generator(val_df, BATCH_SIZE)
    
    # Build model
    model = create_model()
    model.summary()
    
    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_df) // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=len(val_df) // BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1
    )
    
    # Save model
    model.save("steering_model.h5")
    print("Model saved as steering_model.h5")

if __name__ == "__main__":
    main()
