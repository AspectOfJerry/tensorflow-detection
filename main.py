import tensorflow as tf
from tensorflow import keras
from data_loader import load_coco_data
from utils import log, Ccodes

# Constants
dataset_dir = "./dataset"
output_dir = "./output"
category_names = ["cone", "cube"]
input_shape = (756, 756, 3)
num_epochs = 16
batch_size = 8

log(f"Training configuration:"
    f"\n\t- Input size: {input_shape}"
    f"\n\t- Number of epochs: {num_epochs}"
    f"\n\t- Batch size: {batch_size}"
    f"\n\t- Category names: {category_names}"
    f"\n\t- Number of classes: {len(category_names)}", Ccodes.BLUE)

# Create dataset
train_dataset, test_dataset = load_coco_data(dataset_dir, category_names, batch_size)


def custom_model(input_shape, num_classes):
    # Load the MobileNetV3Large base model
    base_model = tf.keras.applications.MobileNetV3Large(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    # Create a new model using the Functional API
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Dense layers for classification
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs, outputs)

    return model


# Create the model
model = custom_model(input_shape, len(category_names))
model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(train_dataset, epochs=num_epochs)

loss, accuracy = model.evaluate(test_dataset)

model.save(output_dir + "/model.h5")
