import dlib
import cv2
import copy
import Loader
import tensorflow as tf
from scipy import misc
import helpers
import time
from datetime import datetime

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list






def run():
    print("Reach Position 1")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fishface=Loader.load_model_fish("googleCKPlus.xml")
    video = cv2.VideoCapture("CindyLRY_ 2019-05-05 17.36.52.mp4")
    # Check if camera opened successfully
    if (video.isOpened() == False):
        print("Error opening video stream or file")
    current = 0
    model = "20180402-114759"
    print("Reach Position 2")
    rectangles=[]
    kkkk=0
    with tf.Graph().as_default():
        # print(tf.get_default_graph())
        print("Reach Position 3")
        with tf.Session() as sess:
            # Load the model

            ## we need to load the model first, then load each layer
            Loader.load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            print("Reach Position 4")
            while (video.isOpened()):

                # videoture frame-by-frame
                try:
                    print("processing frame")
                    print(kkkk)
                    kkkk+=1
                    ret, frame = video.read()

                    height, width, layers = frame.shape
                except AttributeError:
                    break

                size = (width, height)
                frame = helpers.resize(frame, width=1200)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # this is the right place to put the copy,
                # otherwise it will have empty when the face is too big
                temp = copy.deepcopy(frame)

                rects = detector(gray, 1)
                print("Running.....")
                for (i, rect) in enumerate(rects):
                    shape = predictor(gray, rect)
                    shape = helpers.shape_to_np(shape)
                    (x, y, w, h) = helpers.rect_to_coordinate(rect)
                    # draw rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # draw circle
                    for (x1, y1) in shape:
                        cv2.circle(frame, (x1, y1), 2, (0, 0, 255), -1)
                    # to prevent empty frame
                    try:
                        temp = temp[y:y + h, x:x + w]
                        temp_160 = misc.imresize(temp, (160, 160), interp='bilinear')
                        # Snap by the camera save by the time stamp
                        # cv2.imwrite("./camera_photo/{}.png".format(datetime.fromtimestamp(time.time())), temp)
                        # print("SNAP!!!!!!!!!!!!! GIVE A SMILE")
                        if temp_160.ndim == 2:
                            temp_160 = helpers.to_rgb_from2(temp_160);

                        x1, y1, a1 = temp_160.shape
                        temp_re = temp_160.reshape([1, x1, y1, a1])
                        # we put the cropped image to the FaceNet, input shape(1,160,160,3)
                        feed_dict = {images_placeholder: temp_re, phase_train_placeholder: False}
                        # emb return the facial feature of shape (1,512)
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        print("Network running....")

                    except ValueError:
                        pass
                try:
                    tag = helpers.calculation(emb[0])
                    cv2.putText(frame, "{}".format(tag), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print("success!!!!!!1")

                    # out = cv2.resize(temp, (350, 350))
                    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
                    # out=misc.imresize(gray, (350, 350), interp='bilinear')
                    out = cv2.resize(gray, (350, 350))
                    pred, conf = fishface.predict(out)
                    # write on img
                    info1 = 'Guessed emotion: ' + emotions[pred]
                    cv2.putText(frame, info1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 100, 0))
                    print(tag)
                    print(emotions[pred])
                    rectangles.append(frame)


                except( UnboundLocalError, cv2.error):
                    pass

            out = cv2.VideoWriter('ttttttt.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

            for img in rectangles:
                # write to video
                out.write(img)
            out.release()


if __name__ == "__main__":
    run()
