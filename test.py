from tensorflow import keras


def custom_model():
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
        keras.layers.Flatten(),  # 29
        keras.layers.Dense(1024, activation="relu"),  # 29 (fully connected)
        keras.layers.Dense(2, activation="softmax")  # 30 (Softmax), Classifier
    ])

    detection_head = keras.Sequential([
        keras.layers.Conv2D(2, (3, 3), activation="relu", padding="same"),
        # Add more convolutional layers for object detection as needed
    ])

    return model


model = custom_model()

model.summary()
