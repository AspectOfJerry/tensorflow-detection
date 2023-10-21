import tensorflow as tf
from tensorflow import keras

from custom_dataset import CustomDataset
from utils import log, Ccodes

# Constants
DATASET_DIR = "./dataset"
OUTPUT_DIR = "./output"

NUM_CLASSES = 3  # Number of classes, including the background class (+1)
INPUT_SIZE = (3024, 3024)

NUM_EPOCHS = 16
BATCH_SIZE = 32
LABEL_MAP = {
    "background": 0,
    "cube": 1,
    "cone": 2,
}

log(f"Training configuration:"
    f"\n\t- Number of classes: {NUM_CLASSES}"
    f"\n\t- Input size: {INPUT_SIZE}"
    f"\n\t- Number of epochs: {NUM_EPOCHS}"
    f"\n\t- Batch size: {BATCH_SIZE}"
    f"\n\t- Label map: {LABEL_MAP}", Ccodes.BLUE)

train_dataset = CustomDataset(DATASET_DIR, "train", INPUT_SIZE, BATCH_SIZE, LABEL_MAP)

log(f"Number of training images: {len(train_dataset)}", Ccodes.GREEN)


# Define the model architecture
def custom_model_sequential():
    builder = keras.Sequential()

    # Backbone (Convolutional Base)
    builder.add(keras.layers.Input(shape=(*INPUT_SIZE, 3)))

    # Custom convolutional layers
    # TODO: Add custom convolutional layers
    builder.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    builder.add(keras.layers.MaxPooling2D((2, 2)))
    builder.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    builder.add(keras.layers.MaxPooling2D((2, 2)))

    # Region Proposal Network (RPN)
    rpn = keras.Sequential()
    # RPN layers
    # TODO: Add RPN layers
    builder.add(rpn)

    # Detection Head
    detection_head = keras.Sequential()
    # Detection head layers
    # TODO: Add detection head layers
    builder.add(detection_head)

    return builder


# Create the model
model = custom_model_sequential()


# Custom loss function
def custom_loss(y_true, y_pred):
    # TODO: Implement custom loss function
    return  # loss


# Compile the model
model.compile(optimizer="adam", loss=custom_loss)

# Training
model.fit(train_dataset, epochs=NUM_EPOCHS)

# Save the model
model.save(OUTPUT_DIR + "custom_model_sequential.h5")

# Inference
# loaded_model = keras.models.load_model(OUTPUT_DIR + "cortex_model.h5")

# Predict
# predictions = loaded_model.predict(new_data)
