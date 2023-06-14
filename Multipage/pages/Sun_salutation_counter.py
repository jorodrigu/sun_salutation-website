import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from Model.movenet import Movenet
import time
from Model.training import landmarks_to_embedding
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from twilio.rest import Client

account_sid = st.secrets['sid']
auth_token = st.secrets['token']
client = Client(account_sid, auth_token)

token = client.tokens.create()

st.title("Sun Salutation Counter")

count_goal = st.text_input("How many sun salutations do you want to do today?")
line_placeholder = st.empty()

if count_goal:
    line_placeholder.markdown("OK! Let's start with the sun salutations!")


st.title('Yoga Cam')
st.subheader('Start with the sun salutation when you can see your whole body in the video')


#######liveframe-skeleton########

movenet = Movenet('movenet_thunder')
#implement
def detect(input_tensor, inference_count=3):
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)

    for _ in range(inference_count - 1):
        detection = movenet.detect(input_tensor.numpy(),
                                reset_crop_region=False)

    return detection

def coord_landmarks(frame):
    """takes a frame and creates the X values"""
    try:
        image = tf.convert_to_tensor(frame)
    except:
        print(type(frame))
    # image -> landmarks

    person = detect(image)
    min_landmark_score = min([keypoint.score for keypoint in person.keypoints])

    # Get landmarks and scale it to the same size as the input image
    pose_landmarks = np.array(
            [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
            for keypoint in person.keypoints],
                dtype=np.float32)

    # writing the landmark coordinates to its csv files
    coord = pose_landmarks.flatten() #.astype(str).tolist()

    return coord, min_landmark_score
#: np.ndarray | pd.DataFrame
def preprocess_data(X_train):
    X_train = X_train if isinstance(X_train, np.ndarray) else X_train.values
    processed_X_train = []
    for i in range(X_train.shape[0]):
        embedding = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(X_train[i]), (1, 51)))
        processed_X_train.append(tf.reshape(embedding, (34)))
    return tf.convert_to_tensor(processed_X_train)

loaded_model = load_model("Model/BNTB/2023_06_08_BNTB_flipped_img")

category_status = {
    'category0': -1,
    'category1': 0,
    'category2': 0,
    'category3': 0,
    'category4': 0
}

def update_category(category):
    category_status[category] += 1

    # Check if all categories are completed
    if all(category_status.values()) and category == 'category0':
        global counter
        counter += 1
        reset_category_status()
        print('-------------------------------------------')
        print("Counter increased!:", counter)
        print('-------------------------------------------')

# Reset the status of all categories
def reset_category_status():
    for category in category_status:
        category_status[category] = 0

def skelet(frame):
    # Reshape image
    image = tf.convert_to_tensor(frame)
    image = tf.expand_dims(image, axis=0)
    input_image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)

    # Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image, dtype=np.uint8))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])[0, 0]

    img_resized = np.array(input_image).astype(np.uint8)[0]

    keypoints_for_resized = keypoints_with_scores
    keypoints_for_resized[:, 0] *= img_resized.shape[1]
    keypoints_for_resized[:, 1] *= img_resized.shape[0]
    draw_connections(img_resized, keypoints_for_resized, EDGES, 0.3)
    draw_keypoints(img_resized, keypoints_for_resized, 0.3)

    #final_image = cv2.resize(img_resized, (480, 640), interpolation = cv2.INTER_AREA)
    return img_resized, keypoints_for_resized
    # cv2.imshow('Wer das liest ist doof', final_image)

def draw_keypoints(frame, keypoints, confidence_threshold):
    for kp in keypoints:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 2, (0,255,0), 1)

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
def draw_connections(frame, keypoints, edges, confidence_threshold):
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1)


def get_affine_transform_to_fixed_sizes_with_padding(size, new_sizes):
    width, height = new_sizes
    scale = min(height / float(size[1]), width / float(size[0]))
    M = np.float32([[scale, 0, 0], [0, scale, 0]])
    M[0][2] = (width - scale * size[0]) / 2
    M[1][2] = (height - scale * size[1]) / 2
    return M


interpreter = tf.lite.Interpreter(model_path='Model/movenet_thunder.tflite')
interpreter.allocate_tensors()

counter = 0
last_predicted_class=5
# Set the frame rate
fps = 1
# Calculate the frame delay time in seconds
frame_delay =  1/ fps
frame_counter=0
start_time = time.time()
label=""
keypoints_for_resized=None


################################VIDEO####################################
def video_frame_callback(frame):
    image = frame.to_ndarray(format="bgr24")
    global frame_counter
    global start_time
    global frame_delay
    global last_predicted_class
    global label
    global category_status
    global keypoints_for_resized
    global EDGES

    frame_counter+=1
    ##Skip processing for two frames
    # if frame_counter % 10 == 0:

    image, keypoints_for_resized = skelet(image)

    current_time = time.time()
    # If the current time is greater than the start time plus the frame delay
    # Then process the frame and reset the start time
    if current_time > start_time + frame_delay:
        start_time = current_time
    coord, min_landmark_score = coord_landmarks(image)
    # coord2= coord.shape
    coord = coord.reshape((1, 51))
    processed_X_pred = preprocess_data(coord)
    prediction = loaded_model.predict(processed_X_pred, verbose= 0)
    predicted_class = np.argmax(prediction)
    #coord2= coord.shape
    if min_landmark_score >= 0.3:
        #show in which class the prediction is print("Predicted class:", predicted_class)
        if predicted_class != 5:
                if predicted_class != last_predicted_class:
                    if predicted_class == 0:
                        update_category('category0')
                        label = 'You are doing the Mountain!'
                    if predicted_class == 1:
                        update_category('category1')
                        label = 'You are doing the Forward-Bend!'
                    if predicted_class == 2:
                        update_category('category2')
                        label = 'You are doing the Plank!'
                    if predicted_class == 3:
                        update_category('category3')
                        label = 'You are doing the Kobra!'
                    if predicted_class == 4:
                        update_category('category4')
                        label = 'You are doing the Down-Dog!'
                last_predicted_class = predicted_class


    image = cv2.resize(image, (640,640))[80:-80]
    cv2.putText(
            image,
            label,
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    cv2.putText(
            image,
            f"Counter:{counter}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration={  # Add this config
        "iceServers": token.ice_servers},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
