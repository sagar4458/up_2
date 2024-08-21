import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('models/crack_detector.h5')

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or replace with the camera index

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(resized_frame, axis=0)
    img_array = img_array / 255.0

    # Predict if there's a crack
    prediction = model.predict(img_array)

    # Check if crack is detected
    if prediction[0][0] > 0.5:
        cv2.putText(frame, "Crack Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Process the original frame to find the crack and measure its length
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 50, 150)

        # Find contours (which represent cracks in this case)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crack_length = 0
        for contour in contours:
            crack_length += cv2.arcLength(contour, True)
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)  # Draw contour for visualization

        # Convert crack length from pixels to real-world units if calibration data is available
        pixel_to_cm_ratio = 0.1  # Example conversion ratio
        crack_length_cm = crack_length * pixel_to_cm_ratio

        # Display the crack length
        cv2.putText(frame, f"Crack Length: {crack_length_cm:.2f} cm", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Crack Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Crack Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
