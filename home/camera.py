import cv2
import numpy as np

# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model
# from imutils.video import VideoStream
# import imutils
# import cv2,os,urllib.request
# import numpy as np
# from django.conf import settings
#
# import cv2
# import mediapipe as mp
# import time
# import tensorflow as tf
# import numpy as np
# from video_stream import WebcamVideoStream
# from joblib import dump, load
# from PIL import ImageFont, ImageDraw, Image
#
# import tensorflow as tf
# from tensorflow.keras.layers import MaxPool2D ,ReLU,Lambda,TimeDistributed,Dense,\
#     GlobalAveragePooling2D, Dropout,LSTM,Conv2D,MaxPooling2D,Flatten,BatchNormalization
# from keras.models import Model
# from video_stream import WebcamVideoStream
# import numpy as np
# import cv2
#
# number_model = tf.keras.applications.VGG19(weights=None,input_shape=(128, 128,3), classes=10,include_top=False)
#
# flat1 = Flatten()(number_model.layers[-1].output)
#
# class1 = Dense(1024, activation='relu')(flat1)
# # class2 = Dropout(0.2)(class1)
# class3 = Dense(512, activation='relu')(class1)
# # class4 = Dropout(0.2)(class3)
# class5 = Dense(256, activation='relu')(class3)
# # class6 = Dropout(0.2)(class5)
# class7 = Dense(128, activation='relu')(class5)
# # class8 = Dropout(0.2)(class7)
# x = class7
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
# # cap = WebcamVideoStream(0)
# # cap.start()
#
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
# # face_detection_videocam = cv2.CascadeClassifier(os.path.join(
# #             str(settings.BASE_DIR),'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
# # face_detection_webcam = cv2.CascadeClassifier(os.path.join(
# #             str(settings.BASE_DIR),'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
# # # load our serialized face detector model from disk
# # prototxtPath = os.path.sep.join([str(settings.BASE_DIR), "face_detector/deploy.prototxt"])
# # weightsPath = os.path.sep.join([str(settings.BASE_DIR),"face_detector/res10_300x300_ssd_iter_140000.caffemodel"])
# # faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# # maskNet = load_model(os.path.join(str(settings.BASE_DIR),'face_detector/mask_detector.model'))
#
#
#
# number_model.load_weights('model_VGG19_digit.h5')
#
# fe = Model(inputs=number_model.inputs, outputs=number_model.layers[-2].output)
# # fe.summary()
#
# import joblib
#
# RF_model = joblib.load("VGG19_rd_forest_digit.joblib")
#
# bangla_model = tf.keras.applications.Xception(weights=None,input_shape=(71, 71,3), classes=36,include_top=False)
#
# flat1 = Flatten()(bangla_model.layers[-1].output)
#
# class1 = Dense(1024, activation='relu')(flat1)
# # class2 = Dropout(0.2)(class1)
# class3 = Dense(512, activation='relu')(class1)
# # class4 = Dropout(0.2)(class3)
# class5 = Dense(256, activation='relu')(class3)
# # class6 = Dropout(0.2)(class5)
# class7 = Dense(128, activation='relu')(class5)
# # class8 = Dropout(0.2)(class7)
# x = class7
# output = Dense(36, activation='softmax')(x)
# # define new model
# bangla_model = Model(inputs=bangla_model.inputs, outputs=output)
#
# bangla_model.load_weights('model_Xception_characters.h5')
#
# fe_2 = Model(inputs=bangla_model.inputs, outputs=bangla_model.layers[-2].output)
# # fe.summary()
#
# import joblib
#
# RF_model_2 = joblib.load("Xception_rd_forest_characters.joblib")
#
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
#
#
# def number_detection(frame):
#     kernel = np.ones((3, 3), np.uint8)
#
#     # imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = cv2.flip(frame,1)
#     # results = hands.process(imgRGB)
#     # # print(results.multi_hand_landmarks)
#     #
#     # hand_list = findPosition(frame, results)
#     # cropped = False
#     # cropped_images = []
#     # each_part = 0
#     # if len(hand_list)==2:
#     #     bbx_1= hand_list[0]
#     #     bbx_2= hand_list[1]
#     #     x1 = bbx_1[0]
#     #     x2 = bbx_2[0]
#     #
#     #     y1 = bbx_1[3]
#     #     y2 = bbx_2[3]
#     #     if x1<x2:
#     #         start_point = (bbx_1[0],bbx_1[1])
#     #     else:
#     #         start_point = (bbx_2[0],bbx_2[1])
#     #
#     #     if y1<y2:
#     #         end_point = (bbx_2[2],bbx_2[3])
#     #     else:
#     #         end_point = (bbx_1[2],bbx_1[3])
#     cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 0)
#
#     cropped_image = frame[100:400,
#                     100:400]
#
#     try:
#         # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#         # cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 0)
#         albhabets = ["অ", "আ", "ই", "উ", "এ", "ও", "ক", "খ", "গ", "ঘ", "চ", "ছ", "জ", "ঝ", "ট", "ঠ", "ড", "ঢ", "ত",
#                      "থ", "দ", "ধ", "ন", "প", "ফ", "ব", "ভ", "ম", "য়", "র", "ল", "স", "হ", "ড়", "ং", "ঃ"]
#
#         # cropped_image = remove_background(cropped_image)
#         show_image= cropped_image
#         cropped_image = cv2.resize(cropped_image, (71, 71))
#
#         # plt.imshow(img)
#         input_img = np.expand_dims(cropped_image, axis=0)  # Expand dims so the input is (num images, x, y, c)
#         input_img_feature = fe_2.predict(input_img)
#         input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)
#         prediction_RF = RF_model_2.predict(input_img_features)[0]
#         # prediction_RF = le.inverse_transform([prediction_RF])
#
#         print(prediction_RF)
#         # cv2.rectangle(frame, (100,100),
#         #               (255, 0, 0), 3)
#         b, g, r, a = 0, 255, 0, 0
#         fontpath = "Siyamrupali.ttf"
#         font = ImageFont.truetype(fontpath, 48)
#         img_pil = Image.fromarray(frame)
#         draw = ImageDraw.Draw(img_pil)
#         # draw.text((100, 80), u"মুক্তিযুদ্ধ", font = font, fill = (b, g, r, a))
#         predicted= albhabets[prediction_RF]
#
#         draw.text((100, 100), u""+predicted, font=font, fill=(b, g, r, a))
#         img = np.array(img_pil)
#         frame = img
#         # frame = cv2.putText(img, "Pred: " + str(prediction_RF), (100,100),
#         #                     cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#         # frame = show_image
#     except:
#         pass
#
#
#     # for bbox in hand_list:
#     #     cropped = True
#     #     cropped_image = frame[bbox[1] - 20:bbox[3] + 20,
#     #                     bbox[0] - 20:bbox[2] + 20]
#     #
#     #     try:
#     #         # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#     #         # cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 0)
#     #
#     #         cropped_image = remove_background(cropped_image)
#     #         cropped_image = cv2.resize(cropped_image, (64, 64))
#     #
#     #         # plt.imshow(img)
#     #         input_img = np.expand_dims(cropped_image, axis=0)  # Expand dims so the input is (num images, x, y, c)
#     #         input_img_feature = fe_2.predict(input_img)
#     #         input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)
#     #         prediction_RF = RF_model_2.predict(input_img_features)[0]
#     #         # prediction_RF = le.inverse_transform([prediction_RF])
#     #
#     #         print(prediction_RF)
#     #         cv2.rectangle(frame, (bbox[0] - 30, bbox[1] - 30),
#     #                       (bbox[2] + 30, bbox[3] + 30),
#     #                       (255, 0, 0), 3)
#     #         frame = cv2.putText(frame, "Pred: " + str(prediction_RF), (bbox[0] - 10, bbox[1] - 10),
#     #                     cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#     #         # frame = cv2.putText(frame, str(prediction_RF)
#     #         #                     , (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
#     #         #                     (0, 0, 255), 2)
#     #     except Exception as e:
#     #         print(e)
#
#     # if results.multi_hand_landmarks:
#     #     for handLms in results.multi_hand_landmarks:
#     #
#     #         for id, lm in enumerate(handLms.landmark):
#     #             # print(id,lm)
#     #             h, w, c = frame.shape
#     #             cx, cy = int(lm.x * w), int(lm.y * h)
#     #             # if id ==0:
#     #             cv2.circle(frame, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
#     #
#     #         # if len(hand_list):
#     #         #     cv2.rectangle(img, (hand_list[0] - 10, hand_list[1] - 10), (hand_list[2] + 10, hand_list[3] + 10),
#     #         #                   (255, 0, 0), 1)
#     #
#     #         mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
#
#     # define region of interest
#     # roi = frame[100:400, 100:400]
#     return frame
#
#
# def bangla_detection(frame):
#     kernel = np.ones((3, 3), np.uint8)
#     letters = ["০","১","২","৩","৪","৫","৬","৭","৮","৯"]
#
#     imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)
#     # print(results.multi_hand_landmarks)
#
#     hand_list = findPosition(frame, results)
#     cropped = False
#     cropped_images = []
#     each_part = 0
#     for bbox in hand_list:
#         cropped = True
#         cropped_image = frame[bbox[1] - 20:bbox[3] + 20,
#                         bbox[0] - 20:bbox[2] + 20]
#
#         try:
#             # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#             # cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 0)
#             hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
#
#             # define range of skin color in HSV
#             lower_skin = np.array([0, 20, 35], dtype=np.uint8)
#             upper_skin = np.array([20, 255, 255], dtype=np.uint8)
#
#             # extract skin colur imagw
#             mask = cv2.inRange(hsv, lower_skin, upper_skin)
#
#             # extrapolate the hand to fill dark spots within
#             mask = cv2.dilate(mask, kernel, iterations=4)
#
#             # blur the image
#             mask = cv2.GaussianBlur(mask, (1, 1), 100)
#             # find contours
#             contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#             # find contour of max area(hand)
#             try:
#
#                 cnt = max(contours, key=lambda x: cv2.contourArea(x))
#
#                 # approx the contour a little
#                 epsilon = 0.0005 * cv2.arcLength(cnt, True)
#                 approx = cv2.approxPolyDP(cnt, epsilon, True)
#
#                 # make convex hull around hand
#                 hull = cv2.convexHull(cnt)
#
#                 # define area of hull and area of hand
#                 areahull = cv2.contourArea(hull)
#                 areacnt = cv2.contourArea(cnt)
#
#                 # find the percentage of area not covered by hand in convex hull
#                 arearatio = ((areahull - areacnt) / areacnt) * 100
#
#                 # find the defects in convex hull with respect to hand
#                 hull = cv2.convexHull(approx, returnPoints=False)
#                 defects = cv2.convexityDefects(approx, hull)
#
#             except Exception as e:
#                 print(e)
#
#             # l = no. of defects
#             l = 0
#
#             mask = 255 - mask
#             mask_back = mask
#             mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
#             mask = cv2.resize(mask, (128, 128))
#             # plt.imshow(img)
#             input_img = np.expand_dims(mask, axis=0)  # Expand dims so the input is (num images, x, y, c)
#             input_img_feature = fe.predict(input_img)
#             input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)
#             prediction_RF = RF_model.predict(input_img_features)[0]
#             # prediction_RF = le.inverse_transform([prediction_RF])
#
#             print(prediction_RF)
#             cv2.rectangle(frame, (bbox[0] - 30, bbox[1] - 30),
#                           (bbox[2] + 30, bbox[3] + 30),
#                           (255, 0, 0), 3)
#             b, g, r, a = 0, 255, 0, 0
#             fontpath = "Siyamrupali.ttf"
#             font = ImageFont.truetype(fontpath, 48)
#             img_pil = Image.fromarray(frame)
#             draw = ImageDraw.Draw(img_pil)
#             # draw.text((100, 80), u"মুক্তিযুদ্ধ", font = font, fill = (b, g, r, a))
#             predicted = letters[prediction_RF]
#
#             draw.text((bbox[0] - 10, bbox[1] - 10), u"" + predicted, font=font, fill=(b, g, r, a))
#             img = np.array(img_pil)
#             frame = img
#             # frame = cv2.putText(frame, "Pred: " + str(prediction_RF), (bbox[0] - 10, bbox[1] - 10),
#             #             cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#             # frame = cv2.putText(frame, str(prediction_RF)
#             #                     , (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
#             #                     (0, 0, 255), 2)
#         except Exception as e:
#             print(e)
#
#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#
#             for id, lm in enumerate(handLms.landmark):
#                 # print(id,lm)
#                 h, w, c = frame.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 # if id ==0:
#                 cv2.circle(frame, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
#
#             # if len(hand_list):
#             #     cv2.rectangle(img, (hand_list[0] - 10, hand_list[1] - 10), (hand_list[2] + 10, hand_list[3] + 10),
#             #                   (255, 0, 0), 1)
#
#             mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
#
#     # define region of interest
#     # roi = frame[100:400, 100:400]
#     return frame
#
#
#
#
#
# def findPosition(img,results, handNo=0, draw=True):
#     lmlist = []
#
#     bboxes=[]
#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             X_list = []
#             y_list = []
#         # myHand = results.multi_hand_landmarks[handNo]
#             for id, lm in enumerate(handLms.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 lmlist.append([id, cx, cy])
#                 X_list.append(cx)
#                 y_list.append(cy)
#             if len(X_list):
#                 x1 = min(X_list)
#                 x2 = max(X_list)
#                 y1 = min(y_list)
#                 y2 = max(y_list)
#                 bboxes.append([x1,y1,x2,y2])
#             # if draw:
#             #     cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
#
#
#     return bboxes
#
#
# def detection(img):
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)
#     # print(results.multi_hand_landmarks)
#
#     hand_list = findPosition(img, results)
#     cropped = False
#     cropped_images = []
#     each_part = 0
#     for bbox in hand_list:
#         cropped = True
#         cropped_image = img[bbox[1] - 30:bbox[3] + 20,
#                         bbox[0] - 30:bbox[2] + 30]
#         # h = abs(bbox[0]-bbox[2])
#         # w = abs(bbox[1] - bbox[3])
#         # if h > w:
#         #     extra = h - w
#         #     each_part = int(extra/2)
#         #
#         #     cropped_image = img[bbox[1] - (10 + each_part):bbox[3] + (10 + each_part),
#         #                     bbox[0] - 10:bbox[2] + 10]
#         #
#         #     cv2.rectangle(img, (bbox[0] - 10, bbox[1] - (10 + each_part)),
#         #                   (bbox[2] + 10, bbox[3] + (10 + each_part)),
#         #                   (255, 0, 0), 3)
#         #
#         #
#         # else:
#         #     extra = w-h
#         #     each_part = int(extra / 2)
#         #
#         #     cropped_image = img[bbox[1] - 10:bbox[3] + 10,
#         #                     bbox[0] - (10 + each_part):bbox[2] + (10 + each_part)]
#         #     cv2.rectangle(img, (bbox[0] - (10 + each_part), bbox[1] - 10),
#         #                   (bbox[2] + (10 + each_part), bbox[3] + 10),
#         #                   (255, 0, 0), 3)
#         #
#         # l = max([w,h])
#
#         # try:
#         #     cv2.imwrite(image_dir+"/img"+str(count)+".jpg",cropped_image)
#         # except:
#         #     pass
#
#         # count += 1
#         # image_dir
#         # cropped_images.append(cropped_image)
#         try:
#
#             cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#
#             cropped_image = cv2.resize(cropped_image, (28, 28), interpolation=cv2.INTER_AREA)
#             img_array = np.array(cropped_image)
#             img_array = img_array.flatten()
#             img_array = img_array.astype(np.float32())
#             img_array = np.reshape(img_array, (1, 784))
#             img_array = sc.transform(img_array)
#
#             img_new = np.reshape(img_array, (28, 28, 1))
#             # print(img_new.shape)
#             img_array = np.expand_dims(img_new, axis=0)
#             pred = model.predict(img_array)
#             letter = label_map[np.argmax(pred)]
#             print(letter)
#             cv2.rectangle(img, (bbox[0] - 30, bbox[1] - 30),
#                           (bbox[2] + 30, bbox[3] + 30),
#                           (255, 0, 0), 3)
#             cv2.putText(img, "Pred: " + str(letter), (bbox[0] - (10 + each_part), bbox[1] - (10 + each_part)),
#                         cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#         except:
#             pass
#     # print(hand_list)
#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#
#             for id, lm in enumerate(handLms.landmark):
#                 # print(id,lm)
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 # if id ==0:
#                 cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
#
#             # if len(hand_list):
#             #     cv2.rectangle(img, (hand_list[0] - 10, hand_list[1] - 10), (hand_list[2] + 10, hand_list[3] + 10),
#             #                   (255, 0, 0), 1)
#
#             mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
#
#     cTime = time.time()
#     # fps = 1 / (cTime - pTime)
#     pTime = cTime
#
#     # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#     return img
#
# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#
#     def __del__(self):
#         self.video.release()
#
#     def get_frame(self):
#         success, image = self.video.read()
#         # We are using Motion JPEG, but OpenCV defaults to capture raw images,
#         # so we must encode it into JPEG in order to correctly display the
#         # video stream.
#         if success:
#             # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             faces_detected = detection(image)
#             # for (x, y, w, h) in faces_detected:
#             #     cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
#             # frame_flip = cv2.flip(faces_detected,1)
#             ret, jpeg = cv2.imencode('.jpg', faces_detected)
#             return jpeg.tobytes()
#     def stop(self):
#         self.video.release()
#
# class Number_detection(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#
#     def __del__(self):
#         self.video.release()
#
#     def get_frame(self):
#         success, image = self.video.read()
#         # We are using Motion JPEG, but OpenCV defaults to capture raw images,
#         # so we must encode it into JPEG in order to correctly display the
#         # video stream.
#         if success:
#             # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             # faces_detected = detection(image)
#             # for (x, y, w, h) in faces_detected:
#             image = number_detection(image)
#             #     cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
#             # frame_flip = cv2.flip(faces_detected,1)
#             ret, jpeg = cv2.imencode('.jpg', image)
#             return jpeg.tobytes()
#     def stop(self):
#         self.video.release()
#
#
# class bangla_det(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#
#     def __del__(self):
#         self.video.release()
#
#     def get_frame(self):
#         success, image = self.video.read()
#         # We are using Motion JPEG, but OpenCV defaults to capture raw images,
#         # so we must encode it into JPEG in order to correctly display the
#         # video stream.
#
#         # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         faces_detected = bangla_detection(image)
#         # for (x, y, w, h) in faces_detected:
#         #     cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
#         # frame_flip = cv2.flip(faces_detected,1)
#         ret, jpeg = cv2.imencode('.jpg', faces_detected)
#         return jpeg.tobytes()
#     def stop(self):
#         self.video.release()

class test(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # for (x, y, w, h) in faces_detected:
        #     cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        # frame_flip = cv2.flip(faces_detected,1)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    def stop(self):
        self.video.release()