import tensorflow as tf
from tensorflow import keras

from custom_dataset import CustomDataset
from utils import log, Ccodes

# Constants
DATASET_DIR = "./dataset"
OUTPUT_DIR = "./output"

# NUM_CLASSES = 3  # Number of classes, including the background class (+1) <- ?
NUM_CLASSES = 2
INPUT_SHAPE = (224, 224)

NUM_EPOCHS = 16
BATCH_SIZE = 32
LABEL_MAP = {
    "background": 0,
    "cube": 1,
    "cone": 2,
}

log(f"Training configuration:"
    f"\n\t- Number of classes: {NUM_CLASSES}"
    f"\n\t- Input size: {INPUT_SHAPE}"
    f"\n\t- Number of epochs: {NUM_EPOCHS}"
    f"\n\t- Batch size: {BATCH_SIZE}"
    f"\n\t- Label map: {LABEL_MAP}", Ccodes.BLUE)

train_dataset = CustomDataset(DATASET_DIR, "train", INPUT_SHAPE, BATCH_SIZE, LABEL_MAP)

log(f"Number of training images: {len(train_dataset)}", Ccodes.GREEN)


# Define the model architecture (based on MobileNet)
def custom_model(input_shape, num_classes):
    NUM_SCALES = 3
    NUM_ASPECT_RATIOS = 3
    NUM_LOCATIONS = 64 * 64
    NUM_ANCHORS = NUM_SCALES * NUM_ASPECT_RATIOS * NUM_LOCATIONS

    # Backbone (Convolutional Base)
    backbone = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), strides=2, activation="relu", input_shape=(224, 224, 3)),  # 1
        keras.layers.DepthwiseConv2D((3, 3), strides=1, activation="relu", padding="same"),  # 2
        keras.layers.Conv2D(64, (1, 1), strides=1, activation="relu", padding="same"),  # 3
        keras.layers.DepthwiseConv2D((3, 3), strides=2, activation="relu", padding="same"),  # 4
        keras.layers.Conv2D(128, (1, 1), strides=1, activation="relu", padding="same"),  # 5
        keras.layers.DepthwiseConv2D((3, 3), strides=1, activation="relu", padding="same"),  # 6
        keras.layers.Conv2D(128, (1, 1), strides=1, activation="relu", padding="same"),  # 7
        keras.layers.DepthwiseConv2D((3, 3), strides=2, activation="relu", padding="same"),  # 8
        keras.layers.Conv2D(256, (1, 1), strides=1, activation="relu", padding="same"),  # 9
        keras.layers.DepthwiseConv2D((3, 3), strides=1, activation="relu", padding="same"),  # 10
        keras.layers.Conv2D(256, (1, 1), strides=1, activation="relu", padding="same"),  # 11
        keras.layers.DepthwiseConv2D((3, 3), strides=2, activation="relu", padding="same"),  # 12
        keras.layers.Conv2D(512, (1, 1), strides=1, activation="relu", padding="same"),  # 13
        keras.layers.DepthwiseConv2D((3, 3), strides=1, activation="relu", padding="same"),  # 14
        keras.layers.Conv2D(512, (1, 1), strides=1, activation="relu", padding="same"),  # 15
        keras.layers.DepthwiseConv2D((3, 3), strides=1, activation="relu", padding="same"),  # 16
        keras.layers.Conv2D(512, (1, 1), strides=1, activation="relu", padding="same"),  # 17
        keras.layers.DepthwiseConv2D((3, 3), strides=1, activation="relu", padding="same"),  # 18
        keras.layers.Conv2D(512, (1, 1), strides=1, activation="relu", padding="same"),  # 19
        keras.layers.DepthwiseConv2D((3, 3), strides=1, activation="relu", padding="same"),  # 20
        keras.layers.Conv2D(512, (1, 1), strides=1, activation="relu", padding="same"),  # 21
        keras.layers.DepthwiseConv2D((3, 3), strides=1, activation="relu", padding="same"),  # 22
        keras.layers.Conv2D(512, (1, 1), strides=1, activation="relu", padding="same"),  # 23
        keras.layers.DepthwiseConv2D((3, 3), strides=2, activation="relu", padding="same"),  # 24
        keras.layers.Conv2D(1024, (1, 1), strides=1, activation="relu", padding="same"),  # 25
        keras.layers.DepthwiseConv2D((3, 3), strides=1, activation="relu", padding="same"),  # 26
        keras.layers.Conv2D(1024, (1, 1), strides=1, activation="relu", padding="same"),  # 27
        keras.layers.AveragePooling2D((7, 7), strides=1, padding="valid"),  # 28
        keras.layers.Flatten(),  # 29 multi dimensional to linear before passing to dense layer
        keras.layers.Dense(1024, activation="relu"),  # 29 (fully connected)
        keras.layers.Dense(NUM_CLASSES, activation="softmax")  # 30 (Softmax), Classifier
    ])

    detection_head = keras.Sequential([
        keras.layers.Conv2D(num_classes, (3, 3), activation="relu", padding="same"),
        # Add more convolutional layers for object detection as needed
    ])

    # Anchors
    # how the hell am i supposed to do the anchor generation
    anchors = tf.ones((NUM_ANCHORS, 4), dtype=tf.float32)

    input_tensor = keras.layers.Input(shape=input_shape)

    # Connect backbone output to detection head input
    x = backbone(input_tensor)
    detection_output = detection_head(x)

    # Create the complete model
    model = keras.Model(inputs=input_tensor, outputs=[detection_output, anchors])

    return model


# Create the model
# model = custom_model_sequential((224, 224, 3), NUM_CLASSES)
model = custom_model((224, 224, 3), NUM_CLASSES)


# Custom loss function
def custom_loss(y_true, y_pred):
    # TODO: Implement custom loss function
    return  # loss


# Compile the model
model.compile(optimizer="adam")

# Training
model.fit(train_dataset, epochs=NUM_EPOCHS)

# Save the model
model.save(OUTPUT_DIR + "custom_model_sequential.h5")

# Inference
# loaded_model = keras.models.load_model(OUTPUT_DIR + "cortex_model.h5")

# Predict
# predictions = loaded_model.predict(new_data)
