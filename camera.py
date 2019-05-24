import numpy as np
import dlib
import cv2
import copy
import Loader
import tensorflow as tf
from scipy import misc
import pandas as pd
import sys


def calculation(input):
    # print(input)
    # remember to call uploader before calculation, so that we have dataset in 'labeled_pics.csv file'!!!!!!!!!
    data = pd.read_csv('labeled_pics.csv', index_col=0)
    # Since we use the min number to define the best, so for empty we use max, so it will never be chosen
    results = data['features'].apply(
        lambda x: np.sqrt(np.sum(np.square(np.subtract(eval(x), input))) if x != '[]' else sys.float_info.max))
    return results.idxmin()

# extract the four coordinate from the detector
def rect_to_coordinate(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


# convert coordinate to a numpy list
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


# resize the frame to prevent oversize
def resize(image, width=1200):
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def to_rgb_from2(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def to_rgb_from4():
    pass





#     img = facenet.to_rgb(img)
def run():
    print("Reach Position 1")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    video = cv2.VideoCapture(0)
    current = 0
    model = "20180402-114759"
    print("Reach Position 2")
    number = 0
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
            while (True):
                ret, frame = video.read()
                frame = resize(frame, width=1200)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # this is the right place to put the copy,
                # otherwise it will have empty when the face is too big
                temp = copy.deepcopy(frame)

                rects = detector(gray, 1)
                print("Running.....")
                for (i, rect) in enumerate(rects):
                    shape = predictor(gray, rect)
                    shape = shape_to_np(shape)
                    (x, y, w, h) = rect_to_coordinate(rect)
                    # draw rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # draw circle
                    for (x, y) in shape:
                        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                    # to prevent empty frame
                    try:
                        temp = temp[y:y + h , x :x + w ]
                        temp = misc.imresize(temp, (160, 160), interp='bilinear')
                        # When don't wan to snap the picture, comment out the following three lines
                        # cv2.imwrite("name{}.png".format(number), temp)
                        number += 1
                        print("SNAP!!!!!!!!!!!!! GIVE A SMILE")
                        if temp.ndim == 2:
                            temp=to_rgb_from2(temp);
                        # elif temp.ndim==4:
                        #     temp = to_rgb_from4(temp);
                        x1, y1, a1 = temp.shape
                        temp = temp.reshape([1, x1, y1, a1])
                        # we put the cropped image to the FaceNet, input shape(1,160,160,3)
                        feed_dict = {images_placeholder: temp, phase_train_placeholder: False}
                        # emb return the facial feature of shape (1,512)
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        print("Network running....")

                    except ValueError:
                        pass
                try:
                    tag=calculation(emb[0])
                    cv2.putText(frame, "{}".format(tag), (x - 10, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except UnboundLocalError:
                    pass
                # we put the processed frame back to the camera
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                cv2.imshow('frame', rgb)
                # press the key 'q' to quit the program
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # press the key 'p' to snap a picture
                if cv2.waitKey(1) & 0xFF == ord('p'):
                    cv2.imwrite('capture{}.jpg'.format(current), frame)
                    current += 1

    video.release()
    cv2.destroyAllWindows()
run()