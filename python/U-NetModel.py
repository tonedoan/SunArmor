import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd

# Define image size and paths
img_width, img_height = 128, 128  # U-Net often uses 128x128, but you can use larger sizes for better accuracy
train_data_dir = '/kaggle/input/skin-cancer-malignant-vs-benign/train'
test_data_dir = '/kaggle/input/skin-cancer-malignant-vs-benign/test'
batch_size = 16  # You can adjust this based on your memory

# Image data generators for loading and preprocessing images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and testing data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Function to create a U-Net model
def create_unet_model(input_shape):
    inputs = Input(input_shape)

    # Encoder: contracting path
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

   # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder: expansive path
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # Flatten the output for binary classification
    flat = Flatten()(c9)
    dense1 = Dense(128, activation='relu')(flat)
    outputs = Dense(1, activation='sigmoid')(dense1)  # Single output for binary classification

    model = Model(inputs=[inputs], outputs=[outputs])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Function to evaluate the U-Net model
def evaluate_dl_models(train_generator, test_generator):
    input_shape = (img_width, img_height, 3)  # RGB images

    # Create and train the U-Net model
    unet_model = create_unet_model(input_shape)

    # Train the U-Net model
    unet_model.fit(train_generator, epochs=20, validation_data=test_generator,
                   verbose=1)  # Increased epochs for better accuracy

    # Evaluate the model on the test data
    loss, accuracy = unet_model.evaluate(test_generator, verbose=0)

    return {"Model": "U-Net", "Accuracy": accuracy}


# Evaluate the model
dl_results = evaluate_dl_models(train_generator, test_generator)
# Display the result
results = pd.DataFrame([dl_results])
print(results)