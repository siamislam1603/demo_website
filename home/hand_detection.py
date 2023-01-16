import cv2
import mediapipe as mp
import time
import tensorflow as tf
import numpy as np
from video_stream import WebcamVideoStream
from joblib import dump, load

label_map = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
model = tf.keras.models.load_model("sign_model.h5")


sc=load('std_scaler.bin')
cap = WebcamVideoStream(0)
cap.start()

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


def findPosition(img,results, handNo=0, draw=True):
    lmlist = []

    bboxes=[]
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            X_list = []
            y_list = []
        # myHand = results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                X_list.append(cx)
                y_list.append(cy)
            if len(X_list):
                x1 = min(X_list)
                x2 = max(X_list)
                y1 = min(y_list)
                y2 = max(y_list)
                bboxes.append([x1,y1,x2,y2])
            # if draw:
            #     cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)


    return bboxes

image_dir = 'image/'

cropped_win = 0
count =1
while True:
    success, img = cap.read()
    if success:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        hand_list = findPosition(img, results)
        cropped = False
        cropped_images = []
        each_part= 0
        for bbox in hand_list:
            cropped = True
            cropped_image = img[bbox[1] - 30 :bbox[3] + 20,
                                                bbox[0] - 30:bbox[2] + 30]
            # h = abs(bbox[0]-bbox[2])
            # w = abs(bbox[1] - bbox[3])
            # if h > w:
            #     extra = h - w
            #     each_part = int(extra/2)
            #
            #     cropped_image = img[bbox[1] - (10 + each_part):bbox[3] + (10 + each_part),
            #                     bbox[0] - 10:bbox[2] + 10]
            #
            #     cv2.rectangle(img, (bbox[0] - 10, bbox[1] - (10 + each_part)),
            #                   (bbox[2] + 10, bbox[3] + (10 + each_part)),
            #                   (255, 0, 0), 3)
            #
            #
            # else:
            #     extra = w-h
            #     each_part = int(extra / 2)
            #
            #     cropped_image = img[bbox[1] - 10:bbox[3] + 10,
            #                     bbox[0] - (10 + each_part):bbox[2] + (10 + each_part)]
            #     cv2.rectangle(img, (bbox[0] - (10 + each_part), bbox[1] - 10),
            #                   (bbox[2] + (10 + each_part), bbox[3] + 10),
            #                   (255, 0, 0), 3)
            #
            # l = max([w,h])

            # try:
            #     cv2.imwrite(image_dir+"/img"+str(count)+".jpg",cropped_image)
            # except:
            #     pass

            count +=1
            # image_dir
            # cropped_images.append(cropped_image)
            try:

                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                cropped_image = cv2.resize(cropped_image, (28, 28),interpolation = cv2.INTER_AREA)
                img_array = np.array(cropped_image)
                img_array = img_array.flatten()
                img_array = img_array.astype(np.float32())
                img_array = np.reshape(img_array, (1, 784))
                img_array = sc.transform(img_array)

                img_new = np.reshape(img_array, (28, 28, 1))
                # print(img_new.shape)
                img_array = np.expand_dims(img_new, axis=0)
                pred = model.predict(img_array)
                letter = label_map[np.argmax(pred)]
                print(letter)
                cv2.rectangle(img, (bbox[0] - 30 , bbox[1]-30),
                              (bbox[2] + 30 , bbox[3] + 30),
                              (255, 0, 0), 3)
                cv2.putText(img, "Pred: "+str(letter), (bbox[0] - (10+each_part), bbox[1] - (10+each_part)),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            except:
                pass
        # print(hand_list)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:

                for id, lm in enumerate(handLms.landmark):
                    #print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x *w), int(lm.y*h)
                    #if id ==0:
                    cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

                # if len(hand_list):
                #     cv2.rectangle(img, (hand_list[0] - 10, hand_list[1] - 10), (hand_list[2] + 10, hand_list[3] + 10),
                #                   (255, 0, 0), 1)

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime


        cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        print("Total Hand :",len(cropped_images))
        i = 0


        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.stop()
# Destroy all the windows
cv2.destroyAllWindows()