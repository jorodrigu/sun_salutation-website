import streamlit as st

st.set_page_config(layout="wide", page_title='About', page_icon=':sun_with_face:')

st.title('About the Sun Salutation Counter')

st.write('''You want to welcome the Spring season by performing 108 sun salutations (like many yoga practitioners around the world).
         You want the exact number of repetitions since 108 is a sacred number.
         You want to focus on your practice and not on counting.
        Your solution is the sun salutation counter. Perform your sun salutations in front of your webcam and the sun salutation counter does the counting for you''')




st.subheader('This is how the Sun Salutation Counter was built')

'''
- Database for training the model:
    - Kaggle Yoga Pose Image classification dataset & Yoga Poses Dataset
    - Images extracted from sun salutation videos by Le Wagon classmates and teachers


- Extraction of 17 body keypoints with MoveNet Thunder

- Training of the classification model on keypoints of five different poses of the sun salutation:
    - Mountain
    - Forward bend
    - Plank
    - Cobra
    - Downward dog

- Definition of the pose sequence and implementation of the counter'''

st.markdown(f"""<style>
   p {
   background-image: 'sunsal.jpg';
   }</style>""",   unsafe_allow_html=True)
