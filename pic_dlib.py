
import sys
import numpy as np
import dlib
import cv2

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def resize(image, width=1200):
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized






detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
current = 0

image = cv2.imread("WechatIMG10.jpeg")
# frame = resize(image, width=1200)
frame=image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)
for (i, rect) in enumerate(rects):
    print(i)
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)

    (x, y, w, h) = rect_to_bb(rect)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (87, 192, 56), 2)

    cv2.putText(frame, "Triple C #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (87, 192, 56), 2)
    for (x, y) in shape:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)


rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

cv2.imshow('frame', rgb)
# press the key 'q' to quit the program

cv2.imwrite("new.jpeg",frame)
cv2.destroyAllWindows()
# image_file = sys.argv[1]
