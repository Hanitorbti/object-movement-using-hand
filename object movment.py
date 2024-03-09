import cv2
import mediapipe as mp
import numpy as np

mg1 = cv2.imread("D:\pp\photo proccessing\sample.png") 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img,1)
        
        blk = np.zeros((img.shape[0],img.shape[1],3), np.uint8)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for lnd in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img,lnd,mp_hands.HAND_CONNECTIONS)

                middle_finger = lnd.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                middle_finger_x = int(middle_finger.x * img.shape[1] )
                middle_finger_y = int(middle_finger.y * img.shape[0] )

                wrist = lnd.landmark[mp_hands.HandLandmark.WRIST]
                wrist_x = int(wrist.x * img.shape[1] )
                wrist_y = int(wrist.y * img.shape[0] )

                m_x = middle_finger_x + wrist_x
                m_y = middle_finger_y + wrist_y
                m_x = int(m_x/2)
                m_y = int(m_y/2)

                pr1 = blk.shape[0] - abs(middle_finger_y - wrist_y)
                pr2 = blk.shape[1] - abs(middle_finger_x - wrist_x)
                pr = pr1 + pr2
                print(pr)
                if pr > 600 and pr < 1000:
                    pr = int(pr/170)
                else:
                    pr = int(pr/100)

                try:
                    y = int(mg1.shape[0]/pr)
                    x = int(mg1.shape[1]/pr)
                    mg = cv2.resize(mg1, (x,y))
                
                    blk[m_y-(int(y/2)):y+m_y-(int(y/2)), m_x-(int(x/2)):x+m_x-(int(x/2))] = mg
                except:
                    pass

        cv2.imshow('MediaPipe Hands', blk)
        cv2.imshow('MediaPipe Hands2', img)
        if cv2.waitKey(30) & 0xFF == 27:
            break
