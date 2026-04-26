import cv2
import numpy as np


def load_uploaded_image(uploaded_file):
    """
    To convert a Streamlit uploaded file into an OpenCV BGR image.
    """
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode uploaded image.")

    return img


def bgr_to_rgb(image_bgr):
    """
    To convert OpenCV BGR image to RGB for display.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)