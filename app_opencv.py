import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

def preprocess_image(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image

def detect_features(image, max_corners=1000, quality_level=0.01, min_distance=3):
    corners = cv2.goodFeaturesToTrack(image, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    return corners

def calculate_optical_flow_lk(prev_image, next_image, prev_points):
    next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_image, next_image, prev_points, None)
    return next_points, status

def calculate_optical_flow_farneback(prev_image, next_image):
    flow = cv2.calcOpticalFlowFarneback(prev_image, next_image, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
    # flow = cv2.calcOpticalFlowFarneback(prev_image, next_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def visualize_flow(image, prev_points, next_points, status):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    for i, (prev, next) in enumerate(zip(prev_points, next_points)):
        if status[i]:
            plt.arrow(prev[0][0], prev[0][1], next[0][0] - prev[0][0], next[0][1] - prev[0][1], color='red', head_width=3)
    st.pyplot(plt)

def visualize_flow_farneback(image, flow):
    h, w = image.shape
    y, x = np.mgrid[0:h:10, 0:w:10].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.quiver(x, y, fx, fy, color='r', angles='xy', scale_units='xy', scale=1)
    st.pyplot(plt)

# Streamlit app
st.title("Optical Flow for PIV")

# Sidebar for parameters
st.sidebar.header("Optical Flow Parameters")
model = st.sidebar.selectbox("Optical Flow Model", ["Lucas-Kanade", "Farneback"])

if model == "Lucas-Kanade":
    max_corners = st.sidebar.slider("Max Corners", 100, 2000, 1000)
    quality_level = st.sidebar.slider("Quality Level", 0.01, 0.1, 0.01)
    min_distance = st.sidebar.slider("Min Distance", 1, 10, 3)
elif model == "Farneback":
    pyr_scale = st.sidebar.slider("Pyramid Scale", 0.0, 1.0, 0.5)
    levels = st.sidebar.slider("Levels", 1, 10, 3)
    winsize = st.sidebar.slider("Window Size", 5, 50, 15)
    iterations = st.sidebar.slider("Iterations", 1, 10, 3)
    poly_n = st.sidebar.slider("Poly N", 5, 10, 5)
    poly_sigma = st.sidebar.slider("Poly Sigma", 1.1, 2.0, 1.2)
    pass

# Upload images
st.sidebar.header("Upload Images")
image1 = st.sidebar.file_uploader("Upload Image 1", type=["png", "jpg", "jpeg"])
image2 = st.sidebar.file_uploader("Upload Image 2", type=["png", "jpg", "jpeg"])

if image1 and image2:
    prev_image = preprocess_image(load_image(image1))
    next_image = preprocess_image(load_image(image2))

    if model == "Lucas-Kanade":
        prev_points = detect_features(prev_image, max_corners, quality_level, min_distance)
        next_points, status = calculate_optical_flow_lk(prev_image, next_image, prev_points)
        visualize_flow(next_image, prev_points, next_points, status)
    elif model == "Farneback":
        flow = calculate_optical_flow_farneback(prev_image, next_image)
        visualize_flow_farneback(next_image, flow)
