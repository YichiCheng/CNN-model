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
IMAGE_DIR = "C:/image_data/image_data/"
LOG_FILE = "C:/logging_data/log_file_4.txt"  # Adjust the file index as needed
IMG_HEIGHT, IMG_WIDTH = 66, 200  # Resize dimensions
BATCH_SIZE = 32
EPOCHS = 50
AUGMENTATION_PROB = 0.5  # Probability of applying augmentations

def load_data():
    """
    Load the image file paths and corresponding steering angles from the log file.
    """
    data = []
    with open(LOG_FILE, 'r') as file:
        for line in file:
            parts = line.strip().split()
            timestamp, left_frame, right_frame, front_frame, steering_angle = parts
            
            front_frame = f"{front_frame}.jpg"
            left_frame= f"{left_frame}.jpg"
            right_frame= f"{right_frame}.jpg"
            data.append((front_frame, left_frame, right_frame, float(steering_angle)))
    
    df = pd.DataFrame(data, columns=["front_frame", "left_frame", "right_frame", "steering_angle"])
    
    global MIN_ANGLE, MAX_ANGLE  # steering angle normalization
    MIN_ANGLE = df["steering_angle"].min()
    MAX_ANGLE = df["steering_angle"].max()
    print(f"Steering angle range detected: MIN_ANGLE={MIN_ANGLE}, MAX_ANGLE={MAX_ANGLE}")
    # Normalize steering angles
    df["steering_angle"] = (df["steering_angle"] - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE)
    
    return df

def preprocess_image(image_path):
    full_path = os.path.join(IMAGE_DIR, image_path)
    #print(f"Reading image from: {full_path}")  
    img = cv2.imread(full_path)
    if img is None:
        print(f"Image not found or unreadable: {full_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # Resize to match input dimensions
    img = img / 255.0  # Normalize
    return img
 
def data_generator(df, batch_size):
    """
    Generate batches of data for training, including all three cameras.
    """
    num_samples = len(df)
    while True:
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data
        for offset in range(0, num_samples, batch_size):
            batch_samples = df[offset:offset + batch_size]
            front_images = []
            left_images = []
            right_images = []
            angles = []
            
            for _, row in batch_samples.iterrows():
                
                front_image = preprocess_image(row["front_frame"])
                left_image = preprocess_image(row["left_frame"])
                right_image = preprocess_image(row["right_frame"])
                
                steering_angle = row["steering_angle"]
                
                # Append
                front_images.append(front_image)
                left_images.append(left_image)
                right_images.append(right_image)
                angles.append(steering_angle)
            
            yield {
                "front_input": np.array(front_images),
                "left_input": np.array(left_images),
                "right_input": np.array(right_images)
            }, np.array(angles)


# Define CNN model
def create_model():
    """
    Define and compile a CNN model for multi-camera input.
    """
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    front_input = tf.keras.Input(shape=input_shape, name="front_input")
    left_input = tf.keras.Input(shape=input_shape, name="left_input")
    right_input = tf.keras.Input(shape=input_shape, name="right_input")
    
    def cnn_branch(input_layer):
        x = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(input_layer)
        x = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x)
        x = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Flatten()(x)
        return x

    front_branch = cnn_branch(front_input)
    left_branch = cnn_branch(left_input)
    right_branch = cnn_branch(right_input)
    
    # Concatenate features
    concatenated = tf.keras.layers.Concatenate()([front_branch, left_branch, right_branch])
    x = Dense(100, activation='relu')(concatenated)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    output = Dense(1, name="steering_output")(x)
    
    model = tf.keras.Model(inputs=[front_input, left_input, right_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
    return model


# Main function
def main():
    # Load and split data
    df = load_data()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create data generators with augmentation for training and no augmentation for validation
    train_gen = data_generator(train_df, BATCH_SIZE)
    val_gen = data_generator(val_df, BATCH_SIZE)
    
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
    model.save("steering_model_augmented.keras", save_format="keras")
    print("Model saved as steering_model_augmented.keras")

if __name__ == "__main__":
    main()
