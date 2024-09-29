"""
Page segmentation modes:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

"""

"""

OCR engine modes:
  0    Legacy OCR Engine only (--oem 0): Uses the original Tesseract OCR engine, which is based on character pattern recognition.
  1    Neural nets LSTM engine only (--oem 1): Utilizes a Long Short-Term Memory (LSTM) neural network, which is more advanced and generally provides better accuracy.
  2    Legacy + LSTM engines (--oem 2): Combines both the legacy and LSTM engines to leverage the strengths of both.
  3    Default mode (--oem 3): Automatically selects the best available engine based on the input data

"""

import re
import os
import cv2
import pytesseract
import pandas as pd
from PIL import Image
from YOLO_Object_Detection import YOLO_Pred

# Configuration for Tesseract
myconfig = r'--psm 6 --oem 3'

# Paths
onnx_model = './Model/weights/best.onnx'
data_yaml = './data.yaml'
image_path = './thyrocare_0_421.jpg'
output_image_path = 'results/annotated_image.jpg'
output_csv = 'results/extracted_lab_report_box.csv'

# Initialize YOLO model
yolo = YOLO_Pred(onnx_model, data_yaml)

# Load image
image = cv2.imread(image_path)
annotated_image, cropped_images, detected_labels = yolo.predictions(image)

# Save annotated image
cv2.imwrite(output_image_path, annotated_image)
print(f"Annotated image saved to {output_image_path}")

# Clean text function
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\.\-/<>\s]', '', text)
    text = text.replace("O.%", "0.%")
    return text.strip()

# Extract text and append to CSV
def extract_text_and_append_to_csv(cropped_images, detected_labels, output_csv):
    if len(cropped_images) != len(detected_labels):
        print(f"Warning: Number of cropped images ({len(cropped_images)}) does not match number of detected labels ({len(detected_labels)}).")
        return

    data_dict = {
        "Test Name": [],
        "Value": [],
        "Units": [],
        "Reference Range": []
    }

    for cropped_img, label in zip(cropped_images, detected_labels):
        extracted_text = clean_text(pytesseract.image_to_string(cropped_img, config=myconfig))
        print(f"Extracted Text: '{extracted_text}' | Label: '{label}'")  # Debug output

        # Check if label is present
        if label in data_dict:
            # Ensure the extracted text is split correctly
            lines = extracted_text.splitlines()[1:]  # Skip header
            for line in lines:
                if line.strip():
                    print(f"Processing line: {line.strip()}")
                    data_dict[label].append(line.strip())

    # Debugging output for lengths of each category
    for key, value in data_dict.items():
        print(f"{key}: {len(value)} entries")

    # Ensure all lists have the same length before creating DataFrame
    length = min(len(data_dict["Test Name"]), len(data_dict["Value"]), len(data_dict["Units"]), len(data_dict["Reference Range"]))

    # Prepare DataFrame
    df = pd.DataFrame({
        "Test Name": data_dict["Test Name"][:length],
        "Value": data_dict["Value"][:length],
        "Units": data_dict["Units"][:length],
        "Reference Range": data_dict["Reference Range"][:length]
    })

    df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
    print(f"Structured data saved to {output_csv}")

# Run extraction
if cropped_images:
    extract_text_and_append_to_csv(cropped_images, detected_labels, output_csv)
else:
    print("No cropped images found.")