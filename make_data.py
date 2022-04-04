import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
    return img

cap = cv2.VideoCapture(0)
lm_list = []
label = "BODYSWING"
no_of_frames = 600

while True:
    ret, frame = cap.read()
    if ret:
        # Detect objects in the image.
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            frame = draw_landmark_on_image(mpDraw, results, frame)


    cv2.imshow('Mediapipe Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the data
df  = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")

cap.release()
cv2.destroyAllWindows()