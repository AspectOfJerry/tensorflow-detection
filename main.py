import tensorflow as tf
from tensorflow import keras

from custom_dataset import CustomDataset
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


def generate_anchors(input_shape, scales, aspect_ratios, anchor_stride):
    anchors = []

    input_height, input_width = input_shape[:2]

    for y in range(0, input_height, anchor_stride):
        for x in range(0, input_width, anchor_stride):
            for scale in scales:
                for aspect_ratio in aspect_ratios:
                    width = scale * aspect_ratio
                    height = scale / aspect_ratio

                    x_center = x + anchor_stride / 2
                    y_center = y + anchor_stride / 2

                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2

                    anchors.append([x1, y1, x2, y2])

    return tf.constant(anchors, dtype=tf.float32)


# Generate the anchors
scales = [1.0, 2.0]
aspect_ratios = [1.0, 2.0]
anchor_stride = 16
anchors = generate_anchors(INPUT_SHAPE, scales, aspect_ratios, anchor_stride)

# Create a Dataset
dataset = CustomDataset(DATASET_DIR, "train", INPUT_SHAPE, BATCH_SIZE, LABEL_MAP, anchors)
data = dataset.load_data()
data = data.batch(BATCH_SIZE)


# log(f"Number of training images: {len(data)}", Ccodes.GREEN)


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


# def custom_loss(y_true, y_pred):
#     print("y_true shape:", y_true.shape)
#     print("y_pred shape:", y_pred.shape)
#     # Split y_true and y_pred into class labels and bounding box coordinates
#     y_true_boxes, y_true_classes = tf.split(y_true, [3, NUM_CLASSES], axis=-1)
#     y_pred_boxes, y_pred_classes = tf.split(y_pred, [3, NUM_CLASSES], axis=-1)
#
#     print("y_true_boxes shape:", y_true_boxes.shape)
#     print("y_true_classes shape:", y_true_classes.shape)
#     print("y_pred_boxes shape:", y_pred_boxes.shape)
#     print("y_pred_classes shape:", y_pred_classes.shape)
#     # Calculate categorical cross-entropy loss for class labels
#     class_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true_classes, y_pred_classes))
#
#     # Calculate smooth L1 loss for bounding box coordinates
#     diff = y_true_boxes - y_pred_boxes
#     abs_diff = tf.abs(diff)
#     smooth_l1_loss = tf.where(abs_diff < 1, 0.5 * tf.square(diff), abs_diff - 0.5)
#     box_loss = tf.reduce_mean(tf.reduce_sum(smooth_l1_loss, axis=-1))
#
#     # Compute your total loss as a combination of class and box loss (modify weights as needed)
#     total_loss = class_loss + box_loss
#
#     return total_loss

def custom_loss(y_true, y_pred):
    # Split y_true and y_pred into class labels and bounding box coordinates
    y_true_boxes, y_true_classes = tf.split(y_true, [4, NUM_CLASSES], axis=-1)
    y_pred_boxes, y_pred_classes = tf.split(y_pred, [4, NUM_CLASSES], axis=-1)

    # Calculate categorical cross-entropy loss for class labels
    class_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true_classes, y_pred_classes))

    # Calculate smooth L1 loss for bounding box coordinates
    diff = y_true_boxes - y_pred_boxes
    abs_diff = tf.abs(diff)
    smooth_l1_loss = tf.where(abs_diff < 1, 0.5 * tf.square(diff), abs_diff - 0.5)
    box_loss = tf.reduce_mean(tf.reduce_sum(smooth_l1_loss, axis=-1))

    # Compute your total loss as a combination of class and box loss (modify weights as needed)
    total_loss = class_loss + box_loss

    return total_loss


# Compile the models
model.compile(optimizer="adam", loss=custom_loss)

print("Model output shape:", model.output_shape)
# Training
model.fit(data, epochs=NUM_EPOCHS, verbose=1)

# Evaluate the model
# model.evaluate(test_dataset, verbose=1)

# Save the model
model.save(OUTPUT_DIR + "custom_model_sequential.h5")
