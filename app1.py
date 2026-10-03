import streamlit as st
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import time

# Set page configuration
st.set_page_config(page_title="Gesture Talk", layout="wide")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'run' not in st.session_state:
    st.session_state.run = False
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# Custom CSS for styling
st.markdown("""
    <style>
    body { background-color: #f5f5dc; }

    .stButton>button {
        background-color: #deb887;
        color: black;
        font-weight: bold;
        font-size: 18px;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
        box-shadow: 2px 2px 5px #aaa;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #d2b48c;
        color: #2f1e0f;
        transform: scale(1.05);
    }

    .main-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin-top: 30px;
        color: #4b2e2e;
    }

    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #5c4033;
        margin-top: 30px;
        margin-bottom: 40px;
    }

    .emoji-row {
        text-align: center;
        animation: floatY 3s ease-in-out infinite;
        font-size: 40px;
        margin-bottom: 40px;
    }

    @keyframes floatY {
        0% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }

    .prediction-box {
        background-color: #fff8dc;
        border-radius: 10px;
        padding: 10px 20px;
        text-align: center;
        color: #4b2e2e;
        font-weight: bold;
        font-size: 20px;
        box-shadow: 1px 1px 5px #aaa;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Home Page
def homepage():
    st.markdown('<div class="main-title">Gesture Talk</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Bridging silence with technology - Making every gesture count</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="emoji-row">
            🤟 🙌 👋 ✌️ 🙏 🖐️ 👌 👍 👊 👏
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Get Started"):
            st.session_state.page = 'webcam'
            st.session_state.run = False
            st.session_state.last_prediction = None

# Webcam Page
def webcam_page():
    st.markdown('<div class="main-title">Live Sign Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Perform a gesture and get real-time predictions</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not st.session_state.run:
            if st.button("▶️ Start Webcam"):
                st.session_state.run = True
        else:
            if st.button("⏹️ Stop Webcam"):
                st.session_state.run = False

        if st.button("⬅️ Back to Home"):
            st.session_state.page = 'home'
            st.session_state.run = False
            return

        frame_area = st.empty()
        prediction_area = st.empty()

        if st.session_state.run:
            run_webcam(frame_area, prediction_area)

# Run Webcam + Predict
def run_webcam(frame_area, prediction_area):
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
    labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

    cap = cv2.VideoCapture(0)
    offset = 20
    imgSize = 300

    while st.session_state.run:
        success, img = cap.read()
        if not success:
            st.warning("⚠️ Cannot access webcam.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            if imgCrop.size != 0:
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                label = labels[index]
                st.session_state.last_prediction = label

                cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                              (x - offset + 300, y - offset - 10), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, label, (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (0, 255, 0), 4)

        imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        frame_area.image(imgOutput, width=500)

        if st.session_state.last_prediction:
            prediction_area.markdown(
                f"<div class='prediction-box'>🧾 Prediction: {st.session_state.last_prediction}</div>",
                unsafe_allow_html=True)

        time.sleep(0.03)

    cap.release()
    frame_area.empty()
    prediction_area.empty()

# Route based on current page
if st.session_state.page == 'home':
    homepage()
elif st.session_state.page == 'webcam':
    webcam_page()
