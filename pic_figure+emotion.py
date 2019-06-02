import dlib
import cv2
import copy
import Loader
import tensorflow as tf
from scipy import misc
import helpers
import emotion_predictor


emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list


def run(pic,name):
    print("Reach Position 1")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # Check if camera opened successfully

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
            frame=cv2.imread(pic)

            # frame = helpers.resize(frame, width=1200)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # this is the right place to put the copy,
            # otherwise it will have empty when the face is too big


            rects = detector(gray, 1)
            for (i, rect) in enumerate(rects):
                temp = copy.deepcopy(frame)
                data_landmark = []
                shape = predictor(gray, rect)
                shape = helpers.shape_to_np(shape)
                (x, y, w, h) = helpers.rect_to_coordinate(rect)
                # draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (87, 192, 56), 2)

                # draw circle
                for (x1, y1) in shape:
                    cv2.circle(frame, (x1, y1), 4, (0, 0, 255), -1)
                    data_landmark.append(x1)
                    data_landmark.append(y1)
                # to prevent empty frame
                try:
                    temp = temp[y:y + h, x:x + w]
                    print(temp.shape)#this is to trigger error then the temp is out of scale and become empty, we skip it
                    cv2.imwrite("?????{}.jpg".format(name), temp)
                    temp_160 = misc.imresize(temp, (160, 160), interp='bilinear')

                    # Snap by the camera save by the time stamp
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
                    continue
                try:
                    tag = helpers.calculation(emb[0])
                    cv2.putText(frame, "{}".format(tag), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (87, 192, 56), 2)
                    print("success!!!!!!1")

                    # out = cv2.resize(temp, (350, 350))
                    info1=emotion_predictor.output(data_landmark)
                    cv2.putText(frame, info1, (x - 10, y - 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (87, 192, 56),2)
                    print(tag)
                except UnboundLocalError:
                    pass


        out = cv2.imwrite("{}.jpg".format(name),frame)
        print("herer!!!!!!!")




if __name__ == "__main__":
    list_=["./pic_upload/xxx.jpg"]
    name=0
    for video in list_:
        run(video,name)
        name+=1
