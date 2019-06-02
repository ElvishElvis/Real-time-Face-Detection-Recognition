# import dlib
# import cv2
# import copy
# import Loader
# import tensorflow as tf
# from scipy import misc
# import helpers
# import time
# from datetime import datetime
#
# emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
#
#
#
#
#
#
# def run():
#     # print("Reach Position 1")
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#     fishface=Loader.load_model_fish("googleCKPlus.xml")
#     video = cv2.VideoCapture("CindyLRY_ 2019-05-05 17.36.52.mp4")
#     # Check if camera opened successfully
#     if (video.isOpened() == False):
#         print("Error opening video stream or file")
#     current = 0
#     model = "20180402-114759"
#     print("Reach Position 2")
#     rectangles=[]
#     kkkk=0
#
#     print("Reach Position 4")
#     while (video.isOpened()):
#
#         # videoture frame-by-frame
#         try:
#             print("processing frame")
#             print(kkkk)
#             kkkk+=1
#             ret, frame = video.read()
#             print(frame.shape)
#
#             height, width, layers = frame.shape
#         except AttributeError:
#             break
#
#         size = (width, height)
#         # frame = helpers.resize(frame, width=1200)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # this is the right place to put the copy,
#         # otherwise it will have empty when the face is too big
#         temp = copy.deepcopy(frame)
#
#         rects = detector(gray, 1)
#         print("Running.....")
#         for (i, rect) in enumerate(rects):
#             shape = predictor(gray, rect)
#             shape = helpers.shape_to_np(shape)
#             (x, y, w, h) = helpers.rect_to_coordinate(rect)
#             # draw rectangle
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#             # draw circle
#             for (x1, y1) in shape:
#                 cv2.circle(frame, (x1, y1), 2, (0, 0, 255), -1)
#             # to prevent empty frame
#             try:
#                 temp = temp[y:y + h, x:x + w]
#                 print(temp.shape)
#                 temp_160 = misc.imresize(temp, (160, 160), interp='bilinear')
#                 # Snap by the camera save by the time stamp
#                 # cv2.imwrite("./camera_photo/{}.png".format(datetime.fromtimestamp(time.time())), temp)
#                 # print("SNAP!!!!!!!!!!!!! GIVE A SMILE")
#                 if temp_160.ndim == 2:
#                     temp_160 = helpers.to_rgb_from2(temp_160);
#
#                 x1, y1, a1 = temp_160.shape
#                 temp_re = temp_160.reshape([1, x1, y1, a1])
#                 # we put the cropped image to the FaceNet, input shape(1,160,160,3)
#                 # emb return the facial feature of shape (1,512)
#
#             except ValueError:
#                 pass
#         try:
#
#             # out = cv2.resize(temp, (350, 350))
#             gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
#             print(gray.shape)
#             # out=misc.imresize(gray, (350, 350), interp='bilinear')
#             out = cv2.resize(gray, (350, 350))
#             pred, conf = fishface.predict(out)
#             # write on img
#             info1 = 'Guessed emotion: ' + emotions[pred]
#             cv2.putText(frame, info1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 100, 0))
#             rectangles.append(frame)
#
#
#         except(AttributeError):
#             continue
#
#     out = cv2.VideoWriter('ttttttt.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
#
#     for img in rectangles:
#         # write to video
#         out.write(img)
#     out.release()
#
#
# if __name__ == "__main__":
#     run()
su=0
x_list=[3.8,6.58,8.6,11.6,20.1,26.3,10.6,18.35,24.1]
y_list=[4,7,8,20,30,8,2,8,43]
for (x,y) in zip(x_list,y_list):
    su+=(x-y)**2/x
print(su)
