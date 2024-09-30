# Custom Object Character Recognition (OCR) on AWS

## Overview

This project combines YOLOv5 and Tesseract to read specific contents from lab reports. The model is trained to detect relevant text areas and extract information, which is then organized into a structured CSV file. The entire workflow is deployed using Streamlit for a user-friendly interface.

## Features

- **Object Detection**: Utilizes YOLOv5 for real-time object detection in images.
- **Text Extraction**: Employs Tesseract OCR to extract text from detected bounding boxes.
- **Data Organization**: Extracted text is structured into columns for easy analysis.
- **Web Interface**: Built with Streamlit, allowing users to upload images and download extracted data.
- **Deployment Ready**: Can be easily deployed on platforms like AWS.

## Prerequisites

- Python 3.7 or higher
- Libraries: OpenCV, Pandas, NumPy, Streamlit, Tesseract, YOLO Object Detection
- Tesseract OCR installed on your system
- A trained YOLO model in ONNX format

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>