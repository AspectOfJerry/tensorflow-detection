import tensorflow as tf
from tensorflow import keras

from custom_dataset import CustomDataset
from utils import log, Ccodes

# Constants
DATASET_DIR = "./dataset"
OUTPUT_DIR = "./output"

NUM_CLASSES = 3  # Number of classes, including the background class (+1)
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

# Define the model architecture
def custom_model_sequential(input_shape, num_classes):
    NUM_SCALES = 3
    NUM_ASPECT_RATIOS = 3
    NUM_LOCATIONS = 64 * 64
    NUM_ANCHORS = NUM_SCALES * NUM_ASPECT_RATIOS * NUM_LOCATIONS

    # Backbone (Convolutional Base)
    input_tensor = keras.layers.Input(shape=input_shape)
    backbone = keras.applications.ResNet50(include_top=False, input_tensor=input_tensor)

    # Region Proposal Network (RPN)
    rpn_conv = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(backbone.output)
    rpn_class = keras.layers.Conv2D(NUM_ANCHORS, (1, 1), activation="sigmoid", name="rpn_class")(rpn_conv)
    rpn_bbox = keras.layers.Conv2D(NUM_ANCHORS * 4, (1, 1), activation="linear", name="rpn_bbox")(rpn_conv)

    # Detection Head
    detection_head = keras.Sequential([
        keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
    ])

    # Object Detection Output
    detection_class = keras.layers.Conv2D(num_classes, (1, 1), activation="softmax", name="detection_class")(detection_head(rpn_conv))
    detection_bbox = keras.layers.Conv2D(num_classes * 4, (1, 1), name="detection_bbox")(detection_head(rpn_conv))

    # Create the model
    model = keras.Model(inputs=input_tensor, outputs=[rpn_class, rpn_bbox, detection_class, detection_bbox])

    return model


def custom_ssd_lite(input_shape, num_classes):
    # Define the MobileNetV3 backbone
    backbone = keras.applications.MobileNetV3Large(input_shape=input_shape, include_top=False)

    # Feature Pyramid Network (FPN) layers
    x = keras.layers.Conv2D(256, (1, 1), activation="relu", name="fpn_c5p5")(backbone.layers[-1].output)
    x = keras.layers.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(x)
    x = keras.layers.Add(name="fpn_p4add")([x, backbone.layers[-4].output])
    x = keras.layers.Conv2D(256, (1, 1), activation="relu", name="fpn_p4")(x)
    x = keras.layers.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(x)
    x = keras.layers.Add(name="fpn_p3add")([x, backbone.layers[-7].output])
    x = keras.layers.Conv2D(256, (1, 1), activation="relu", name="fpn_p3")(x)
    x = keras.layers.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(x)
    x = keras.layers.Add(name="fpn_p2add")([x, backbone.layers[-10].output])
    x = keras.layers.Conv2D(256, (1, 1), activation="relu", name="fpn_p2")(x)

    # SSD Head
    num_anchors = 3 * 3  # NUM_SCALES * NUM_ASPECT_RATIOS
    x = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="ssd_head")(x)
    classification = keras.layers.Conv2D(num_anchors * num_classes, (3, 3), padding="same", name="classification")(x)
    regression = keras.layers.Conv2D(num_anchors * 4, (3, 3), padding="same", name="regression")(x)

    # Create the model
    model = keras.Model(inputs=backbone.input, outputs=[classification, regression])

    return model


# Create the model
# model = custom_model_sequential((224, 224, 3), NUM_CLASSES)
model = custom_ssd_lite((3024, 3024, 3), NUM_CLASSES)


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
