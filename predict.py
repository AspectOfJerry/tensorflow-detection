import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

image_size = 224
num_classes = 3


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
    y_true_boxes, y_true_classes = tf.split(y_true, [4, num_classes], axis=-1)
    y_pred_boxes, y_pred_classes = tf.split(y_pred, [4, num_classes], axis=-1)

    # Print the shapes of the split tensors
    tf.print("y_true_boxes shape:", tf.shape(y_true_boxes))
    tf.print("y_true_classes shape:", tf.shape(y_true_classes))
    tf.print("y_pred_boxes shape:", tf.shape(y_pred_boxes))
    tf.print("y_pred_classes shape:", tf.shape(y_pred_classes))

    # Compute categorical cross-entropy loss for the class labels
    class_loss = keras.losses.categorical_crossentropy(y_true_classes, y_pred_classes)

    # Compute smooth L1 loss for the bounding box coordinates
    box_loss = keras.losses.huber(y_true_boxes, y_pred_boxes)

    # Print the losses
    tf.print("class_loss:", class_loss)
    tf.print("box_loss:", box_loss)

    # Average the losses
    total_loss = tf.reduce_mean(class_loss + box_loss)

    return total_loss


# Load the model, specifying the custom loss function
model = tf.keras.models.load_model('outputcustom_model_sequential.h5', custom_objects={'custom_loss': custom_loss})

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame for the model
    # This might need to be adjusted depending on how your model expects the input
    input_image = cv2.resize(frame, (image_size, image_size))
    input_image = input_image / 255.0  # normalize to [0,1]
    input_image = np.expand_dims(input_image, axis=0)  # add batch dimension

    # Make a prediction
    prediction = model.predict(input_image)

    # Postprocess the prediction
    # This might need to be adjusted depending on how your model outputs the prediction
    boxes, classes = np.split(prediction, [4], axis=-1)

    # Draw the bounding boxes on the frame
    for box, cls in zip(boxes[0], classes[0]):
        xmin, ymin, xmax, ymax = box
        # Convert to absolute coordinates
        xmin = int(xmin * frame.shape[1])
        ymin = int(ymin * frame.shape[0])
        xmax = int(xmax * frame.shape[1])
        ymax = int(ymax * frame.shape[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, str(cls), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Real-time predictions", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
