import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np

# Load trained YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Function to blur plates in an image
def blur_license_plates(image):
    results = model(image)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            plate = image[y1:y2, x1:x2]
            if plate.size != 0:
                blurred = cv2.GaussianBlur(plate, (51, 51), 30)
                image[y1:y2, x1:x2] = blurred
    return image

# Streamlit GUI
st.title("License Plate Blurring App")
st.write("Upload an image or video, and this app will automatically detect and blur license plates.")

option = st.radio("Choose input type:", ("Image", "Video"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        blurred_img = blur_license_plates(img.copy())

        # Show side by side
        st.image([cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                  cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)],
                 caption=["Original", "Blurred"],
                 width=500)

        # Encode blurred image to bytes for download
        _, buffer = cv2.imencode(".jpg", blurred_img)
        st.download_button(
            label="Download Blurred Image",
            data=buffer.tobytes(),
            file_name="blurred_image.jpg",
            mime="image/jpeg"
        )

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("output.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            blurred_frame = blur_license_plates(frame)
            out.write(blurred_frame)

        cap.release()
        out.release()

        st.video("output.mp4")
        with open("output.mp4", "rb") as file:
            st.download_button("Download Blurred Video", file, "blurred_video.mp4")

