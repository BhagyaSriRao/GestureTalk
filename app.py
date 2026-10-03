import streamlit as st
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import time

st.markdown(
    """
    <style>
    body {
        background-color: #f5f5dc;
    }
    .stButton>button {
        background-color: #deb887;
        color: #000000;
        font-weight: 600;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 24px;
        border: none;
        box-shadow: 2px 2px 5px #aaa;
    }
    .stButton>button:hover {
        background-color: #d2b48c;
        color: #2f1e0f;
    }
    </style>
    """, unsafe_allow_html=True
)

if 'run' not in st.session_state:
    st.session_state.run = False

def run_webcam():
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
    labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

    cap = cv2.VideoCapture(0)

    offset = 20
    imgSize = 300

    # Create a placeholder for video frames
    frame_placeholder = st.empty()
    prediction_placeholder = st.empty()

    while st.session_state.run:
        success, img = cap.read()
        if not success:
            st.warning("Failed to capture image from webcam.")
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

                cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                              (x - offset + 300, y - offset - 10), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (0, 255, 0), 4)

                prediction_placeholder.markdown(f"<h3 style='color: #704214;'>🧾 Prediction: {labels[index]}</h3>", unsafe_allow_html=True)
            else:
                prediction_placeholder.empty()
        else:
            prediction_placeholder.empty()

        imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(imgOutput, use_column_width=True)

        # Control frame rate
        time.sleep(0.03)

    cap.release()
    frame_placeholder.empty()
    prediction_placeholder.empty()

def main():
    st.title("🤟 Gesture Talk - \n Bridging Silence With Technology")

    if not st.session_state.run:
        if st.button("Start Webcam"):
            st.session_state.run = True
    else:
        if st.button("Stop Webcam"):
            st.session_state.run = False

    if st.session_state.run:
        run_webcam()

main()
