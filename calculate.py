import numpy as np
import dlib
import cv2
import copy
import Loader
import tensorflow as tf
import camera
from scipy import misc

##return the list of 512 feature & the list of face matrix
def calculate_feature(img):
    with tf.Graph().as_default():
        feature_list = []
        img_list = []
        img = cv2.imread(img)
        img = camera.resize(img, width=1200)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # this is the right place to put the copy,
        # otherwise it will have empty when the face is too big
        rects = detector(gray, 1)
        with tf.Session() as sess:
            # Load the model

            ## we need to load the model first, then load each layer
            Loader.load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            print("Running.....")
            for (i, rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = camera.shape_to_np(shape)
                (x, y, w, h) = camera.rect_to_coordinate(rect)
                try:
                    img = img[y :y + int(h / 2), x :x + int(w / 2)]
                    img = misc.imresize(img, (160, 160), interp='bilinear')

                    x1, y1, a1 = img.shape
                    # when 4 dimension
                    temp = copy.deepcopy(img)
                    temp = temp.reshape([1, x1, y1, a1])
                    # we put the cropped image to the FaceNet, input shape(1,160,160,3)
                    feed_dict = {images_placeholder: temp, phase_train_placeholder: False}
                    # emb return the facial feature of shape (1,512)
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    feature_list.append(emb.flatten())
                    img_list.append(img.flatten())

                except ValueError:
                    print("error")
                    return None

    return feature_list, img_list
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

model = "20180402-114759"
# print('hreee')


a,b=calculate_feature("wirter.jpg")
print(len(a))
print(len(b))
print(b[0].shape)
print(a[0].shape)
