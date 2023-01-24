# import mediapipe as mp
# import time
# from joblib import dump, load
# from PIL import ImageFont, ImageDraw, Image
#
# import tensorflow as tf
# from tensorflow.keras.layers import MaxPool2D ,ReLU,Lambda,TimeDistributed,Dense,\
#     GlobalAveragePooling2D, Dropout,LSTM,Conv2D,MaxPooling2D,Flatten,BatchNormalization
# from keras.models import Model
# import numpy as np
# import cv2
#
# # this VGG19 model is used for detecting letter from hand mask
# number_model = tf.keras.applications.VGG19(weights=None,input_shape=(128, 128,3), classes=10,include_top=False) # model intialization
#
# flat1 = Flatten()(number_model.layers[-1].output)
#
#
# class1 = Dense(1024, activation='relu')(flat1)  # this is the FIRST layer of the network and it has 1024 neurons
# class3 = Dense(512, activation='relu')(class1)  # this is the SECOND layer of the network and it has 512 neurons
# class5 = Dense(256, activation='relu')(class3)  # this is the THIRD layer of the network and it has 256 neurons
# class7 = Dense(128, activation='relu')(class5)  # this is the FOURTH layer of the network and it has 128 neurons
# x = class7
# # this layer has 10 layer because we have 10 classes to predict
# output = Dense(10, activation='softmax')(x)
# # define new model
# number_model = Model(inputs=number_model.inputs, outputs=output)
#
#
# label_map = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# model = tf.keras.models.load_model("sign_model.h5")
#
#
# sc=load('std_scaler.bin')
#
# # mediapipe library is used to detect the hand gesture
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(static_image_mode=False,
#                       max_num_hands=2,
#                       min_detection_confidence=0.5,
#                       min_tracking_confidence=0.5)
# mpDraw = mp.solutions.drawing_utils
#
# pTime = 0
# cTime = 0
#
# # loading the pretrained model
# number_model.load_weights('model_VGG19_digit.h5')
#
# # This is the second detection model
# fe = Model(inputs=number_model.inputs, outputs=number_model.layers[-2].output)
#
#
# import joblib
#
# # This is the second detection model (trained) loading
# RF_model = joblib.load("VGG19_rd_forest_digit.joblib")
#
#
# # this Xception model is used to detect the bangla character from hand gesture
# bangla_model = tf.keras.applications.Xception(weights=None,input_shape=(71, 71,3), classes=36,include_top=False)
#
# flat1 = Flatten()(bangla_model.layers[-1].output)
#
# # first neural network layer has 1024 neurons
# class1 = Dense(1024, activation='relu')(flat1)
# class3 = Dense(512, activation='relu')(class1)
# class5 = Dense(256, activation='relu')(class3)
# class7 = Dense(128, activation='relu')(class5)
# x = class7
# # this layer has 36 layer because we have 36 classes to predict
# output = Dense(36, activation='softmax')(x)
# # define new model
# bangla_model = Model(inputs=bangla_model.inputs, outputs=output)
#
# # loading the pretrained model
# bangla_model.load_weights('model_Xception_characters.h5')
#
# fe_2 = Model(inputs=bangla_model.inputs, outputs=bangla_model.layers[-2].output)
#
# import joblib
#
# # loading the second pretrained model
# RF_model_2 = joblib.load("Xception_rd_forest_characters.joblib")

# def remove_background(img):
#     # img1 = rgb2gray(img)
#     img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     img2 = img
#     # blur = cv2.GaussianBlur(img1,(3,3),0)
#     # ret,thresh1 = cv2.threshold(img1,225,255,cv2.THRESH_BINARY_INV)
#     thresh1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 133, 5)
#
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
#     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#     b, g, r = cv2.split(img2)
#     for i in range(closing.shape[0]):
#         for j in range(closing.shape[1]):
#             if (closing[i, j]) == 0:
#                 b[i, j] = 0
#                 g[i, j] = 0
#                 r[i, j] = 0
#
#     newimg = cv2.merge((b, g, r))
#     return newimg

# this function is for detecting bangla alphabet from images
def character_detection(frame):
    kernel = np.ones((3, 3), np.uint8)

    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 0)

    # this is tha roi
    cropped_image = frame[100:400,
                    100:400]

    try:
        # intializing the alphabet list
        albhabets = ["অ", "আ", "ই", "উ", "এ", "ও", "ক", "খ", "গ", "ঘ", "চ", "ছ", "জ", "ঝ", "ট", "ঠ", "ড", "ঢ", "ত",
                     "থ", "দ", "ধ", "ন", "প", "ফ", "ব", "ভ", "ম", "য়", "র", "ল", "স", "হ", "ড়", "ং", "ঃ"]

        # resizing image to pass into the model perfectly
        cropped_image = cv2.resize(cropped_image, (71, 71))

        input_img = np.expand_dims(cropped_image, axis=0)  # Expand dims so the input is (num images, x, y, c)
        input_img_feature = fe_2.predict(input_img) # predicting the alphabet via first model
        input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)
        prediction_RF = RF_model_2.predict(input_img_features)[0] # predicting the alphabet via second model
        b, g, r, a = 0, 255, 0, 0
        fontpath = "Siyamrupali.ttf"
        font = ImageFont.truetype(fontpath, 48)
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        predicted= albhabets[prediction_RF]

        draw.text((100, 100), u""+predicted, font=font, fill=(b, g, r, a)) # typing the alphabet on image
        img = np.array(img_pil)
        frame = img
    except:
        pass

    return frame


# this function is for detecting bangla numbers from images
def bangla_detection(frame):
    kernel = np.ones((3, 3), np.uint8)
    letters = ["০","১","২","৩","৪","৫","৬","৭","৮","৯"]

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # finding the hand position from the frame
    hand_list = findPosition(frame, results)
    cropped = False
    cropped_images = []
    each_part = 0
    for bbox in hand_list:
        cropped = True
        # making the roi based on the detected hand position
        cropped_image = frame[bbox[1] - 20:bbox[3] + 20,
                        bbox[0] - 20:bbox[2] + 20]

        try:
            # this block of codes make hand mask

            hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

            # define range of skin color in HSV
            lower_skin = np.array([0, 20, 35], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            # extract skin colur imagw
            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # extrapolate the hand to fill dark spots within
            mask = cv2.dilate(mask, kernel, iterations=4)

            # blur the image
            mask = cv2.GaussianBlur(mask, (1, 1), 100)
            # find contours
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # find contour of max area(hand)
            try:

                cnt = max(contours, key=lambda x: cv2.contourArea(x))

                # approx the contour a little
                epsilon = 0.0005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # make convex hull around hand
                hull = cv2.convexHull(cnt)

                # define area of hull and area of hand
                areahull = cv2.contourArea(hull)
                areacnt = cv2.contourArea(cnt)

                # find the percentage of area not covered by hand in convex hull
                arearatio = ((areahull - areacnt) / areacnt) * 100

                # find the defects in convex hull with respect to hand
                hull = cv2.convexHull(approx, returnPoints=False)
                defects = cv2.convexityDefects(approx, hull)

            except Exception as e:
                print(e)

            l = 0

            mask = 255 - mask # inversing the mask
            # mask_back = mask
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) # converting mask to rgb image
            mask = cv2.resize(mask, (128, 128))
            input_img = np.expand_dims(mask, axis=0)  # Expand dims so the input is (num images, x, y, c)
            input_img_feature = fe.predict(input_img)
            input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)
            prediction_RF = RF_model.predict(input_img_features)[0]

            # drawing the rectangle
            cv2.rectangle(frame, (bbox[0] - 30, bbox[1] - 30),
                          (bbox[2] + 30, bbox[3] + 30),
                          (255, 0, 0), 3)
            b, g, r, a = 0, 255, 0, 0
            fontpath = "Siyamrupali.ttf"
            font = ImageFont.truetype(fontpath, 48)
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            # draw.text((100, 80), u"মুক্তিযুদ্ধ", font = font, fill = (b, g, r, a))
            predicted = letters[prediction_RF]

            draw.text((bbox[0] - 10, bbox[1] - 10), u"" + predicted, font=font, fill=(b, g, r, a))
            img = np.array(img_pil)
            frame = img
        except Exception as e:
            print(e)

    # this block of code draws hand landmarks
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # if id ==0:
                cv2.circle(frame, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    return frame



# this function finds the hand position from the frame

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


    return bboxes


# this function detects the english sign language
def english_detection(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    hand_list = findPosition(img, results)
    cropped = False
    each_part = 0
    for bbox in hand_list:
        cropped = True
        cropped_image = img[bbox[1] - 30:bbox[3] + 20,
                        bbox[0] - 30:bbox[2] + 30]

        try:

            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            cropped_image = cv2.resize(cropped_image, (28, 28), interpolation=cv2.INTER_AREA)
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
            cv2.rectangle(img, (bbox[0] - 30, bbox[1] - 30),
                          (bbox[2] + 30, bbox[3] + 30),
                          (255, 0, 0), 3)
            cv2.putText(img, "Pred: " + str(letter), (bbox[0] - (10 + each_part), bbox[1] - (10 + each_part)),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        except:
            pass
    # this block of code draws hand landmarks
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # if id ==0:
                cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    return img


# this is a class which calls the english alphabet detection function and provide frame to the webpage
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        if success:
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces_detected = english_detection(image)
            # for (x, y, w, h) in faces_detected:
            #     cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            # frame_flip = cv2.flip(faces_detected,1)
            ret, jpeg = cv2.imencode('.jpg', faces_detected)
            return jpeg.tobytes()
    def stop(self):
        self.video.release()

# this is a class which calls the alphabet detection function and provide frame to the webpage
class Number_detection(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        if success:
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # faces_detected = detection(image)
            # for (x, y, w, h) in faces_detected:
            image = character_detection(image)
            #     cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            # frame_flip = cv2.flip(faces_detected,1)
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
    def stop(self):
        self.video.release()

# this is a class which calls the number detection function and provide frame to the webpage

class bangla_det(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

        faces_detected = bangla_detection(image)
        ret, jpeg = cv2.imencode('.jpg', faces_detected)
        return jpeg.tobytes()
    def stop(self):
        self.video.release()

# this is not used in anywhere. this was coded for testing only
class test(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    def stop(self):
        self.video.release()