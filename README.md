# **License Plate Blurring App**

A simple web app built with Streamlit and YOLO that automatically detects and blurs vehicle license plates in images and videos.

### **Features**

* Upload images or videos

* Automatic detection of license plates using a trained YOLO model

* Smart Gaussian blur applied to detected plates

* Download the blurred result directly from the browser

### **Motivation**

I've always been interested in machine learning and computer vision, so I decided to create a small but practical project that uses AI to solve a real-world privacy problem.

### **Installation**

1. Clone this repository:

   git clone https://github.com/yourusername/license-plate-blur.git

   cd license-plate-blur

3. Install dependencies:

   pip install -r requirements.txt

5. Run the app:

   streamlit run app.py

### **Tech stack**

Python

Streamlit - for the web interface

Ultralytics YOLO - for object detection

OpenCV - for image processing and blurring

