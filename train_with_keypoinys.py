import cv2
import mediapipe as mp
import numpy as np
import os
import time
import csv

A_path = os.path.join("data", "A")
B_path = os.path.join("data", "B")
C_path = os.path.join("data", "C")

folder = 'data_new/A'
keypoints_folder = 'keypoints/A.csv'
label = folder.split('/')[-1]
label_dict = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
}

label_ = label_dict[label]
counter = 0

cap = cv2.VideoCapture(0)
import math
mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hand.Hands(
    static_image_mode = False,
    max_num_hands = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.5

)



while cap.isOpened():

    ret,frame = cap.read()

    if not ret:
        print("Error in grabbing frame")
        break
    
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    if results.multi_hand_landmarks:

        for hand_landmark in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmark, mp_hand.HAND_CONNECTIONS)
                h, w, _ = frame.shape
                coords_2d = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmark.landmark])
                # print(coords_2d.shape)

        x,y,w,h = cv2.boundingRect(coords_2d)  
        cv2.rectangle(frame,(x - 20,y - 20),(x+w + 20,y+h + 20),(0,255,0),2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = frame[y - 20:y + h + 20, x - 20:x + w + 20]
        cv2.imshow("Cropped Hand", img)
        imgWhite =  np.ones((300,300,3),np.uint8)*255
        
        aspect_ratio = h/w

        if aspect_ratio >1:
            k =  300/h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(img,(wCal, 300))


        else:  
            k =  300/w
            hCal = math.ceil(k * h)   
            imgResize = cv2.resize(img,(300, hCal))
           
              


        imgWhite[(300 - imgResize.shape[0])//2:(300 + imgResize.shape[0])//2,(300 - imgResize.shape[1])//2:(300 + imgResize.shape[1])//2] = imgResize
        cv2.imshow("imagewhite", imgWhite)



        # finger tips points
        finger_tips_indices = [0,4,8,12,16,20]
        finger_tips  = coords_2d[finger_tips_indices]
        finger_tips = finger_tips - finger_tips[0]
        finger_tips = finger_tips[1:]

        #normalise the points by the area of the bounding box
        area = w * h
        finger_tips = finger_tips / area

    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('s'):
         counter += 1
         cv2.imwrite(f'{folder}/Image{time.time()}.jpg',imgWhite)


         with open(keypoints_folder,'a',newline='') as f:
     
             writer = csv.writer(f)
             row = [label_] + finger_tips.flatten().tolist()
             writer.writerow(row)
         print(counter)

cap.release()
cv2.destroyAllWindows()

