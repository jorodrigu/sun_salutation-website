# import csv
import pandas as pd
from tensorflow import keras
# from sklearn.model_selection import train_test_split
from Model.data import BodyPart
import tensorflow as tf
# from keras import models
# from keras import optimizers
# import matplotlib.pyplot as plt


tfjs_model_dir = 'model'


# loading final csv file
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.drop(['filename'],axis=1, inplace=True)
    classes = df.pop('class_name').unique()
    y = df.pop('class_no')

    X = df.astype('float64')
    y = keras.utils.to_categorical(y)

    return X, y, classes


def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""
    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5
    return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.

    It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
    """
    # Hips center
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                 BodyPart.RIGHT_HIP)

    # Shoulders center
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                      BodyPart.RIGHT_SHOULDER)

    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center - hips_center)
    # Pose center
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                     BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(pose_center_new,
                                    [tf.size(landmarks) // (17*2), 17, 2])

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
    return pose_size



def normalize_pose_landmarks(landmarks):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
    scaling it to a constant pose size.
  """
  # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                 BodyPart.RIGHT_HIP)

    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center,
                                [tf.size(landmarks) // (17*2), 17, 2])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size
    return landmarks


def landmarks_to_embedding(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(17, 3)
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])
    # Flatten the normalized landmark coordinates into a vector
    embedding = keras.layers.Flatten()(landmarks)
    return embedding


def preprocess_data(X_train):
    processed_X_train = []
    for i in range(X_train.shape[0]):
        embedding = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(X_train.iloc[i]), (1, 51)))
        processed_X_train.append(tf.reshape(embedding, (34)))
    return tf.convert_to_tensor(processed_X_train)

# def plot_history(history, title='', axs=None, exp_name=""):
#     if axs is not None:
#         ax1, ax2 = axs
#     else:
#         f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

#     if len(exp_name) > 0 and exp_name[0] != '_':
#         exp_name = '_' + exp_name
#     ax1.plot(history.history['loss'], label = 'train' + exp_name)
#     ax1.plot(history.history['val_loss'], label = 'val' + exp_name)
#     ax1.set_ylim(0., 2.2)
#     ax1.set_title('loss')
#     ax1.legend()

#     ax2.plot(history.history['accuracy'], label='train accuracy'  + exp_name)
#     ax2.plot(history.history['val_accuracy'], label='val accuracy'  + exp_name)
#     ax2.set_ylim(0.25, 1.)
#     ax2.set_title('Accuracy')
#     ax2.legend()
#     plt.show()
#     return (ax1, ax2)

# if __name__ == "__main__":
#     X, y, class_names = load_csv('Model/train_data.csv')
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
#     X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5)
#     #X_test, y_test, class_names = load_csv('test_data.csv')

#     processed_X_train = preprocess_data(X_train)
#     processed_X_val =  preprocess_data(X_val)
#     processed_X_test = preprocess_data(X_test)

#     reg_l1 = keras.regularizers.L1(0.01)
#     reg_l2 = keras.regularizers.L2(0.01)


#     model = keras.Sequential([
#         keras.layers.Dense(128, activation=tf.nn.relu6, input_shape=(34,)),
#         keras.layers.Dropout(0.5),
#         keras.layers.Dense(64, activation=tf.nn.relu6, ), #bias_regularizer=reg_l2),
#         keras.layers.Dropout(0.5),
#         keras.layers.Dense(32, activation=tf.nn.relu6),
#         keras.layers.Dropout(0.2),
#         keras.layers.Dense(len(class_names), activation="softmax")
#     ])

#     #adam_opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

#     model.compile(
#         optimizer='adam',
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     # Add a checkpoint callback to store the checkpoint that has the highest
#     # validation accuracy.
#     checkpoint_path = "weights.best.hdf5"
#     checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                 monitor='val_accuracy',
#                                 verbose=0,
#                                 save_best_only=True,
#                                 mode='max')
#     earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
#                                                 patience=20)
#     tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs')

#     # Start training
#     print('--------------TRAINING----------------')
#     history = model.fit(processed_X_train, y_train,
#                         epochs=500,
#                         batch_size=32,
#                         verbose=1,
#                         validation_data=(processed_X_val, y_val),
#                         callbacks=[checkpoint, earlystopping, tensorboard_callback])

#     plot_history(history, title='', axs=None, exp_name="")

#     print('-----------------EVAUATION----------------')
#     loss, accuracy = model.evaluate(processed_X_test, y_test)
#     print('LOSS: ', loss)
#     print("ACCURACY: ", accuracy)


    # print('-----------------PREDICTION OF A NEW PIC----------------')
    # X_pred, y_pred, class_names= load_csv('Model/predict_data.csv')
    # processed_X_pred = preprocess_data(X_pred)
    # prediction= model.predict(processed_X_pred)
    # print('prediciton: ', prediction)

    #terminal:tensorboard --logdir=./logs

    #tfjs.converters.save_keras_model(model, tfjs_model_dir)
    #print('tfjs model saved at ',tfjs_model_dir)

    # models.save_model(model, "Model/BNTB/2023_06_08_BNTB_flipped_img")
