import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet152
import pandas as pd

# Define image size and paths
img_width, img_height = 224, 224  # ResNet typically uses 224x224 input size
train_data_dir = '/kaggle/input/skin-cancer-malignant-vs-benign/train'
test_data_dir = '/kaggle/input/skin-cancer-malignant-vs-benign/test'
batch_size = 32

# Image data generators for loading and preprocessing images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and testing data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'  # Binary classification: benign vs malignant
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)


# Function to create the ResNet152 model
def create_resnet152_model(input_shape):
    base_model = ResNet152(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model layers to prevent retraining
    base_model.trainable = False

    # Build the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to evaluate the ResNet152 model
def evaluate_dl_models(train_generator, test_generator):
    input_shape = (img_width, img_height, 3)  # RGB images

    # Create and train the ResNet152 model
    resnet_model = create_resnet152_model(input_shape)

    # Train the model
    resnet_model.fit(train_generator, epochs=20, validation_data=test_generator, verbose=1)

    # Evaluate the model on the test data
    loss, accuracy = resnet_model.evaluate(test_generator, verbose=0)

    return {"Model": "ResNet152", "Accuracy": accuracy}


# Evaluate the model
dl_results = evaluate_dl_models(train_generator, test_generator)

# Display the result
results = pd.DataFrame([dl_results])
print(results)