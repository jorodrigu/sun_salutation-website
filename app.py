import streamlit as st
import cv2

st.title("Sun Salutation Counter")

count_goal = st.text_input("How many sun salutations do you want to do today?")
line_placeholder = st.empty()

if count_goal:
    line_placeholder.markdown("OK! Let's start with the sun salutations!")


st.title('Yoga Cam')
st.subheader('Start with the sun salutation when you can see your whole body in the video')

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
# Capture frame-by-frame
    ret, frame = cap.read()
    print(frame)
