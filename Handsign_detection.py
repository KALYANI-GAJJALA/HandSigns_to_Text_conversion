import cv2
import mediapipe as mp
import pandas as pd
import pickle
import time

with open("hand_sign_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

vid = cv2.VideoCapture(4)

output_text = ""
is_writing_enabled = False
latest_gesture = " "
confirmed_gesture = " "
gesture_count = 0
threshold = 5  # Number of consistent frames

last_write_time = time.time()
write_interval = 2  
line_spacing = 35
text_start_y = 120

img = cv2.imread(r"C:\Users\kalya\OneDrive\Desktop\Hand_sign-project\hand_signs.jpg")
img = cv2.resize(img, (500, 500))

while True:
    succ, frame = vid.read()
    if succ == False:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_name = "None"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            if len(coords) == 63:
                df = pd.DataFrame([coords], columns=[str(i) for i in range(63)])
                gesture_name = model.predict(df)[0]

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if gesture_name == latest_gesture:
        gesture_count += 1
    else:
        latest_gesture = gesture_name
        gesture_count = 1

    if gesture_count >= threshold:
        confirmed_gesture = latest_gesture
        gesture_count = 0  

        if confirmed_gesture == "start":
            is_writing_enabled = True
        elif confirmed_gesture == "stop":
            is_writing_enabled = False

        current_time = time.time()
        if is_writing_enabled and (current_time - last_write_time >= write_interval):
            if confirmed_gesture == "delete":
                output_text = output_text[:-1]
            elif confirmed_gesture == "space":
                output_text += " "
            elif confirmed_gesture == "next line":
                output_text += "\n"
            elif confirmed_gesture not in ["start", "stop", "None"]:
                output_text += confirmed_gesture

            last_write_time = current_time

   
    cv2.putText(frame, f"Gesture: {confirmed_gesture}", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    status = "Writing Mode: ON" if is_writing_enabled else "Writing Mode: OFF"
    status_color = (0, 255, 0) if is_writing_enabled else (0, 0, 255)
    cv2.putText(frame, status, (10, 80),cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    y = text_start_y
    for line in output_text.split("\n"):
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y += line_spacing

    cv2.imshow("Gesture to Text", frame)
    cv2.imshow("Hand Signs", img)

    if cv2.waitKey(1) & 255 == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
