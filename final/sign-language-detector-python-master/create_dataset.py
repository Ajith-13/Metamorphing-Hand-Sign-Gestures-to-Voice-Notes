import os
import pickle

import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = 'A:/finalpro/final/data'

data = []
labels = []

# Iterate through directories and files in DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    # Check if dir_path is a directory
    if os.path.isdir(dir_path):
        # Iterate through files in the directory
        for img_path in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_path)
            # Check if img_full_path is a file
            if os.path.isfile(img_full_path):
                data_aux = []
                x_ = []
                y_ = []
                img = cv2.imread(img_full_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            x_.append(x)
                            y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

# Save data and labels to a pickle file
with open('A:/finalpro/final/data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
