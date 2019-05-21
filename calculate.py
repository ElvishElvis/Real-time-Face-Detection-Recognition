import numpy as np
import dlib
import cv2
import copy
import Loader
import tensorflow as tf
import camera
from scipy import misc

##return the list of 512 feature & the list of face matrix
def calculate_feature(imgs):
    img_list = []
    ppp=0

    with tf.Graph().as_default():
        with tf.Session() as sess:
            ## we need to load the model first, then load each layer
            Loader.load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            for img in imgs:
                img = cv2.imread(img)
                img = camera.resize(img, width=1200)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # this is the right place to put the copy,
                # otherwise it will have empty when the face is too big
                rects = detector(gray, 1)

                print("Running.....")
                for (i, rect) in enumerate(rects):
                    shape = predictor(gray, rect)
                    shape = camera.shape_to_np(shape)
                    (x, y, w, h) = camera.rect_to_coordinate(rect)
                    try:
                        img = img[y :y + h , x :x +w ]
                        print(img.shape)
                        print('img.shape')
                        img = misc.imresize(img, (160, 160), interp='bilinear')
                        cv2.imwrite("name{}.jpg".format(ppp),img)
                        ppp+=1
                        print(img.shape)
                        print('img.reshape')
                        # x1, y1, a1 = img.shape
                        # # when 4 dimension
                        # temp = copy.deepcopy(img)
                        # temp = temp.reshape([1, x1, y1, a1])
                        img_list.append(img)

                    except ValueError:
                        print("error")
                        break;
            all_img=np.stack(img_list, axis=0)
            # we put the cropped image to the FaceNet, input shape(1,160,160,3)
            feed_dict = {images_placeholder: all_img, phase_train_placeholder: False}
            # emb return the facial feature of shape (1,512)
            embs = sess.run(embeddings, feed_dict=feed_dict)

    return  all_img ,embs

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


model = "20180402-114759"


imgs=["Molly003.jpeg","Molly002.jpeg","Molly001.jpeg","Molly004.jpeg"]

a,b=calculate_feature(imgs)
print(a.shape) # (4, 160, 160, 3)
print(b.shape) # (4, 512)

#(4, 160, 160, 3)-> a[0][0]=a[1][0]=a[2][0]=a[3][0](160,160,3)