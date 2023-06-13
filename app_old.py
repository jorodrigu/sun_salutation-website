import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

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


#Just to show the video on the streamlit page
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    def transform(self, frame):
        img = frame.to_ndarray(format='bgr24')
        return img

def main():
    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.width = 800
        webrtc_ctx.video_transformer.height = 600

    if webrtc_ctx.video_transformer:
        st.write('Processed video:')
        st.image(webrtc_ctx.video_transformer.frame_out)

if __name__ == "__main__":
    main()
