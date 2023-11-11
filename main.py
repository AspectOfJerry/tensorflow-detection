import tensorflow as tf
from tensorflow import keras

from custom_dataset import CustomDataset
from utils import log, Ccodes

# Constants
DATASET_DIR = "./dataset"
OUTPUT_DIR = "./output"
NUM_CLASSES = 3
LABEL_MAP = {"background": 0, "cube": 1, "cone": 2}
INPUT_SHAPE = (224, 224, 3)
NUM_EPOCHS = 16
BATCH_SIZE = 8
NUM_PREDICTIONS = 9

log(f"Training configuration:"
    f"\n\t- Number of classes (including background): {NUM_CLASSES}"
    f"\n\t- Input size: {INPUT_SHAPE}"
    f"\n\t- Number of epochs: {NUM_EPOCHS}"
    f"\n\t- Batch size: {BATCH_SIZE}"
    f"\n\t- Label map: {LABEL_MAP}", Ccodes.BLUE)

# Create dataset
dataset = CustomDataset(DATASET_DIR, "train", INPUT_SHAPE, BATCH_SIZE, LABEL_MAP, NUM_PREDICTIONS)


def custom_model(input_shape, num_classes, num_predictions=9):
    # Backbone (Convolutional Base)
    base_model = tf.keras.applications.EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    # Detection head
    # detection_head = keras.Sequential([
    #     keras.layers.Conv2D(num_classes + 4, (3, 3), activation="relu", padding="same"),
    #     keras.layers.Reshape((-1, num_classes + 4))
    # ])
    # Detection head
    detection_head = keras.Sequential([
        keras.layers.Conv2D(num_predictions * (num_classes + 4), (3, 3), activation="relu", padding="same"),
        keras.layers.Reshape((-1, num_predictions, num_classes + 4))
    ])

    # Connect backbone output to detection head input
    x = base_model.output
    detection_output = detection_head(x)

    # Create the complete model
    model = keras.Model(inputs=base_model.input, outputs=detection_output)

    return model


# Create the model
model = custom_model(INPUT_SHAPE, NUM_CLASSES, NUM_PREDICTIONS)

model.summary()


def custom_loss(y_true, y_pred):
    print("------------------------------------------------------------------------------")
    print("Debugging custom loss function:")

    y_true_class = y_true[..., :NUM_CLASSES]
    y_true_bbox = y_true[..., NUM_CLASSES:]
    y_pred_class = y_pred[..., :NUM_CLASSES]
    y_pred_bbox = y_pred[..., NUM_CLASSES:]

    # Print the shapes before the binary crossentropy calculation
    print("BEFORE binary crossentropy:")
    print("y_true_class shape:", y_true_class.shape)
    print("y_true_bbox shape:", y_true_bbox.shape)
    print("y_pred_class shape:", y_pred_class.shape)
    print("y_pred_bbox shape:", y_pred_bbox.shape)

    print("------------------------------------------------------------------------------")

    class_loss = tf.keras.losses.BinaryCrossentropy()(y_true_class, y_pred_class)
    box_loss = tf.keras.losses.Huber()(y_true_bbox, y_pred_bbox)
    total_loss = class_loss + box_loss

    # Print the shapes after the binary crossentropy calculation
    print("AFTER binary crossentropy:")
    print("y_true_class shape:", y_true_class.shape)
    print("y_true_bbox shape:", y_true_bbox.shape)
    print("y_pred_class shape:", y_pred_class.shape)
    print("y_pred_bbox shape:", y_pred_bbox.shape)

    print("------------------------------------------------------------------------------")

    print("class_loss:", class_loss)
    print("box_loss:", box_loss)
    print("total_loss:", total_loss)

    return total_loss


# Compile the model
model.compile(optimizer="adam", loss=custom_loss)

# Training loop
for epoch in range(NUM_EPOCHS):
    for batch_idx in range(len(dataset)):
        images, y_true = dataset[batch_idx]

        # Training step
        model.train_on_batch(images, y_true)

# Evaluate the model on the training data
train_loss = model.evaluate(dataset, verbose=1)
print("Train loss:", train_loss)

# Save the model
model.save(OUTPUT_DIR + "custom_model_sequential.h5")
