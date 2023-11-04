import tensorflow as tf
from tensorflow import keras

from custom_dataset import create_dataset
from utils import log, Ccodes

# Constants
DATASET_DIR = "./dataset"
OUTPUT_DIR = "./output"
NUM_CLASSES = 3
LABEL_MAP = {
    "background": 0,
    "cube": 1,
    "cone": 2
}
INPUT_SHAPE = (224, 224, 3)
NUM_EPOCHS = 16
BATCH_SIZE = 32
NUM_ANCHORS = 9  # The maximum number of objects that can be detected in an image

log(f"Training configuration:"
    f"\n\t- Number of classes (including background): {NUM_CLASSES}"
    f"\n\t- Input size: {INPUT_SHAPE}"
    f"\n\t- Number of epochs: {NUM_EPOCHS}"
    f"\n\t- Batch size: {BATCH_SIZE}"
    f"\n\t- Label map: {LABEL_MAP}", Ccodes.BLUE)

train_dataset = create_dataset(DATASET_DIR, "train", INPUT_SHAPE, BATCH_SIZE, LABEL_MAP, NUM_ANCHORS, NUM_CLASSES)
# test_dataset = create_dataset(DATASET_DIR, "test", INPUT_SHAPE, BATCH_SIZE, LABEL_MAP, NUM_ANCHORS, NUM_CLASSES)

log(f"Number of training images: {len(train_dataset)}", Ccodes.GREEN)


def generate_anchors(input_shape, scales, aspect_ratios):
    anchors = []

    input_height, input_width = input_shape[:2]

    for scale in scales:
        for aspect_ratio in aspect_ratios:
            width = scale * aspect_ratio
            height = scale / aspect_ratio

            x_center = input_width / 2
            y_center = input_height / 2

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            anchors.append([x1, y1, x2, y2])

    return tf.constant(anchors, dtype=tf.float32)


# Generate the anchors
scales = [1.0, 2.0]
aspect_ratios = [1.0, 2.0]
anchors = generate_anchors(INPUT_SHAPE, scales, aspect_ratios)


# Define the model architecture (based on MobileNet)
def custom_model(input_shape, num_classes, anchors):
    num_anchors = len(anchors)

    # Backbone (Convolutional Base)
    backbone = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), strides=2, activation="relu", input_shape=input_shape),  # 1
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
    ])

    # Detection head
    detection_head = keras.Sequential([
        keras.layers.Conv2D(num_anchors * (num_classes + 4), (3, 3), activation="relu", padding="same"),
        keras.layers.Reshape((-1, num_classes + 4))
    ])

    # Connect backbone output to detection head input
    input_tensor = keras.layers.Input(shape=input_shape)
    x = backbone(input_tensor)
    detection_output = detection_head(x)

    # Create the complete model
    model = keras.Model(inputs=input_tensor, outputs=detection_output)

    return model


# Create the model
model = custom_model(INPUT_SHAPE, NUM_CLASSES, anchors)
print("Model created")


# Author: basically GitHub Copilot
def custom_loss(y_true, y_pred):
    # Expand the dimensions of y_true to make it a rank 3 tensor
    y_true = tf.expand_dims(y_true, axis=1)

    # Print the shapes of y_true and y_pred
    tf.print("y_true shape:", tf.shape(y_true))
    tf.print("y_pred shape:", tf.shape(y_pred))

    # Determine the maximum number of bounding boxes
    max_boxes = tf.reduce_max([tf.shape(y_true)[1], tf.shape(y_pred)[1]])

    # Pad y_true and y_pred with zeros so they have the same number of boxes
    y_true = tf.pad(y_true, [[0, 0], [0, max_boxes - tf.shape(y_true)[1]], [0, 0]])
    y_pred = tf.pad(y_pred, [[0, 0], [0, max_boxes - tf.shape(y_pred)[1]], [0, 0]])

    # Print the shapes of y_true and y_pred after padding
    tf.print("y_true shape after padding:", tf.shape(y_true))
    tf.print("y_pred shape after padding:", tf.shape(y_pred))

    # Split y_true and y_pred into class labels and bounding box coordinates
    y_true_boxes, y_true_classes = tf.split(y_true, [4, NUM_CLASSES], axis=-1)
    y_pred_boxes, y_pred_classes = tf.split(y_pred, [4, NUM_CLASSES], axis=-1)

    # Print the shapes of the split tensors
    tf.print("y_true_boxes shape:", tf.shape(y_true_boxes))
    tf.print("y_true_classes shape:", tf.shape(y_true_classes))
    tf.print("y_pred_boxes shape:", tf.shape(y_pred_boxes))
    tf.print("y_pred_classes shape:", tf.shape(y_pred_classes))

    # Compute categorical cross-entropy loss for the class labels
    class_loss = keras.losses.categorical_crossentropy(y_true_classes, y_pred_classes)

    # Compute smooth L1 loss for the bounding box coordinates
    box_loss = keras.losses.huber(y_true_boxes, y_pred_boxes)

    # Create a mask for the non-empty bounding boxes in y_true
    mask = tf.reduce_any(y_true != 0, axis=-1)

    # Apply the mask to the losses
    class_loss = tf.boolean_mask(class_loss, mask)
    box_loss = tf.boolean_mask(box_loss, mask)

    # Print the losses
    tf.print("class_loss:", class_loss)
    tf.print("box_loss:", box_loss)

    # Average the losses
    total_loss = tf.reduce_mean(class_loss + box_loss)

    return total_loss


# Compile the models
model.compile(optimizer="adam", loss=custom_loss)

# Training
model.fit(train_dataset, epochs=NUM_EPOCHS, verbose=1)

# Evaluate the model
# model.evaluate(test_dataset, verbose=1)

# Save the model
model.save(OUTPUT_DIR + "custom_model_sequential.h5")
