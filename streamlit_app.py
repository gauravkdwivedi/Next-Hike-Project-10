import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
from YOLO_Object_Detection import YOLO_Pred
import pandas as pd
import re

# Configuration for Tesseract
myconfig = r'--psm 6 --oem 3'

# Load YOLO Model
st.title("Custom OCR and YOLO Object Detection")
onnx_model = './Model/weights/best.onnx'
data_yaml = './data.yaml'

# Initialize YOLO model
yolo = YOLO_Pred(onnx_model, data_yaml)

# Upload image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Clean text function
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\.\-/<>\s]', '', text)
    text = text.replace("O.%", "0.%")
    return text.strip()

# Extract text and append to CSV
def extract_text_and_append_to_df(cropped_images, detected_labels):
    data_dict = {
        "Test Name": [],
        "Value": [],
        "Units": [],
        "Reference Range": []
    }

    for cropped_img, label in zip(cropped_images, detected_labels):
        extracted_text = clean_text(pytesseract.image_to_string(cropped_img, config=myconfig))

        # Check if label is present
        if label in data_dict:
            # Ensure the extracted text is split correctly
            lines = extracted_text.splitlines()[1:]  # Skip header
            for line in lines:
                if line.strip():
                    data_dict[label].append(line.strip())

    # Prepare DataFrame
    df = pd.DataFrame({
        "Test Name": data_dict["Test Name"],
        "Value": data_dict["Value"],
        "Units": data_dict["Units"],
        "Reference Range": data_dict["Reference Range"]
    })

    return df

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # YOLO Prediction
    annotated_image, cropped_images, detected_labels = yolo.predictions(img_array)

    # Display results
    st.image(annotated_image, caption='Annotated Image with Detections', use_column_width=True)
    
    if cropped_images:
        # Extract text and create a DataFrame
        df = extract_text_and_append_to_df(cropped_images, detected_labels)

        # Display extracted data
        st.write("Extracted Lab Report Data")
        st.dataframe(df)

        # Option to download CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Extracted Data as CSV", csv, "extracted_lab_report.csv", "text/csv")
    else:
        st.write("No objects detected.")