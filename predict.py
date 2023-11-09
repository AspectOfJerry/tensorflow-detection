import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

image_size = 224
num_classes = 3


def custom_loss(y_true, y_pred):
    # Separate the class labels and the bounding box coordinates in y_true
    y_true_class = y_true[..., :num_classes]
    y_true_box = y_true[..., num_classes:]

    # Separate the class predictions and the bounding box predictions in y_pred
    y_pred_class = y_pred[..., :num_classes]
    y_pred_box = y_pred[..., num_classes:]

    # Compute binary cross-entropy loss for the classes
    class_loss = tf.keras.losses.BinaryCrossentropy()(y_true_class, y_pred_class)

    # Compute smooth L1 loss for the bounding boxes
    box_loss = tf.keras.losses.Huber()(y_true_box, y_pred_box)

    # Compute the total loss
    total_loss = class_loss + box_loss

    return total_loss


# Load the model, specifying the custom loss function
model = tf.keras.models.load_model("outputcustom_model_sequential.h5", custom_objects={"custom_loss": custom_loss})

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame for the model
    input_image = cv2.resize(frame, (image_size, image_size))
    input_image = input_image / 255.0  # normalize to [0,1]
    input_image = np.expand_dims(input_image, axis=0)  # add batch dimension

    # Make a prediction
    prediction = model.predict(input_image)[0]
    print(prediction)

    # Postprocess the prediction
    boxes, classes = np.split(prediction, [4], axis=-1)  # might need to change this

    # Draw the bounding boxes on the frame
    for box, cls in zip(boxes, classes):
        xmin, ymin, xmax, ymax = box
        xmin, ymin, xmax, ymax = map(int, [xmin * frame.shape[1], ymin * frame.shape[0], xmax * frame.shape[1], ymax * frame.shape[0]])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Get the class with the highest probability
        class_id = np.argmax(cls)
        cv2.putText(frame, str(class_id), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Print the class and coordinates
        # print(f"Class: {class_id}, Coordinates: {(xmin, ymin, xmax, ymax)}")

    # Display the frame
    cv2.imshow("Real-time predictions", frame)

    # Break the loop if "q" is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
