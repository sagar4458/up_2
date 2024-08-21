import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="models/crack_detector.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# GStreamer pipeline for capturing video from a camera
# Adjust the pipeline string based on your camera's specifications and requirements
gst_pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw, width=640, height=480, framerate=30/1 ! "
    "videoconvert ! appsink"
)

# Initialize the camera with GStreamer pipeline
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Unable to open camera with GStreamer.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video frame.")
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(resized_frame, axis=0)
    img_array = img_array / 255.0
    img_array = img_array.astype(np.float32)

    # Set the tensor for the input data
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run the inference
    interpreter.invoke()

    # Get the prediction result
    prediction = interpreter.get_tensor(output_details[0]['index'])

    if prediction[0][0] > 0.5:
        cv2.putText(frame, "Crack Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray_frame, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank mask
        mask = np.zeros_like(frame)

        crack_length = 0
        for contour in contours:
            crack_length += cv2.arcLength(contour, True)

            # Draw the contour on the mask
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Apply the mask to the original frame
        masked_frame = cv2.bitwise_and(frame, mask)

        # Convert crack length from pixels to centimeters
        pixel_to_cm_ratio = 0.1  # Replace with your calibrated ratio
        crack_length_cm = crack_length * pixel_to_cm_ratio

        # Display the crack length
        cv2.putText(masked_frame, f"Crack Length: {crack_length_cm:.2f} cm", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        masked_frame = frame.copy()
        cv2.putText(masked_frame, "No Crack Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Crack Detection', masked_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
