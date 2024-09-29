import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        """
        Initialize the YOLO model and load class labels from a YAML file.
        :param onnx_model: Path to the YOLO ONNX model file.
        :param data_yaml: Path to the YAML file containing class names and other configurations.
        """
        # Load the YAML file (which contains class labels and number of classes)
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        # Extract class labels and number of classes from the YAML file
        self.labels = data_yaml['names']  # Class labels
        self.nc = data_yaml['nc']         # Number of classes
        
        # Load the YOLO model from ONNX format
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        # Use OpenCV backend and set the target device (CPU)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self, image):
        """
        Perform object detection on the input image using the YOLO model.
        :param image: Input image in which objects need to be detected.
        :return: Tuple of (annotated image with bounding boxes, list of cropped images for detected regions).
        """
        
        row, col, d = image.shape  # Get dimensions of the image (height, width, channels)

        # Convert the image into a square array for YOLO processing
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)  # Create a blank square image
        input_image[0:row, 0:col] = image  # Place the original image into this square image

        # Prepare the image for YOLO input: scale the image and create a "blob"
        input_width_yolo = 640  # YOLOv5 model typically uses 640x640 input size
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (input_width_yolo, input_width_yolo), swapRB=True, crop=False)
        self.yolo.setInput(blob)  # Pass the blob as input to YOLO model
        
        # Get predictions from the YOLO model
        preds = self.yolo.forward()

        # Non-Maximum Suppression (NMS) - filter detections by confidence and class scores
        detections = preds[0]  # YOLO model returns predictions in a specific format
        boxes = []             # To store bounding boxes
        confidences = []       # To store confidence values
        classes = []           # To store class IDs
        cropped_images = []    # To store cropped images of detected regions

        # Image dimensions scaling factors (to map boxes back to original image size)
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / input_width_yolo
        y_factor = image_h / input_width_yolo

        # Loop through each detection in the predictions
        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]  # Extract confidence (objectness score)
            
            # If the confidence is above a certain threshold (e.g., 0.4)
            if confidence > 0.4:
                # Get the class ID with the highest probability
                class_score = row[5:].max()
                class_id = row[5:].argmax()

                # Only consider detections where the class score is above the threshold (e.g., 0.25)
                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]  # Center x, center y, width, height of the bounding box
                    
                    # Convert center coordinates to top-left corner and scale the box to original image size
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    # Append the box, confidence, and class ID to their respective lists
                    box = np.array([left, top, width, height])
                    boxes.append(box)
                    confidences.append(confidence)
                    classes.append(class_id)

        # Convert lists to NumPy arrays for NMS processing
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()

        # Initialize the list for detected labels
        detected_labels = []

        # Draw the filtered bounding boxes on the image and crop detected regions
        for ind in index:
            # Get the coordinates and confidence of the detected object
            x, y, w, h = boxes_np[ind]
            bb_conf = int(confidences_np[ind] * 100)  # Convert confidence to percentage
            classes_id = classes[ind]  # Get the class ID
            class_name = self.labels[classes_id]  # Get the corresponding class name
            colors = self.generate_colors(classes_id)  # Get color for the bounding box based on class ID

            # Prepare the label text with class name and confidence percentage
            text = f'{class_name}: {bb_conf}%'

            # Draw the bounding box and label on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), colors, 2)
            cv2.rectangle(image, (x, y - 30), (x + w, y), colors, -1)  # Background for text label
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)
            
            # Crop the detected region
            cropped_img = image[y:y+h, x:x+w]
            cropped_images.append(cropped_img)
            detected_labels.append(class_name)  # Append class name to the detected labels list

        # Return the annotated image, cropped regions and detected_labels
        return image, cropped_images, detected_labels
    
    def generate_colors(self, ID):
        """
        Generate unique colors for each class ID for drawing bounding boxes.
        :param ID: The class ID for which the color is being generated.
        :return: A tuple containing RGB values for the color.
        """
        np.random.seed(10)  # Set random seed for consistent color generation
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()  # Generate random colors
        return tuple(colors[ID])  # Return the color corresponding to the class ID