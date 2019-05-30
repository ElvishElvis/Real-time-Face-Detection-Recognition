
import sys
import numpy as np
import dlib
import cv2
import emotion_predictor
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





video = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
current = 0


while (True):
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    data=[]
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, "Michchdskjfakjsbdfjkabsjkdfhkqe #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)
            data.append(x)
            data.append(y)
    tag=emotion_predictor.output(data)
    print(tag)
    cv2.putText(frame, "tag{}".format(tag), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)


    cv2.imshow('frame', rgb)
    # press the key 'q' to quit the program


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # press the key 'p' to take a picture
    if cv2.waitKey(1) & 0xFF == ord('p'):
        cv2.imwrite('capture{}.jpg'.format(current), frame)
        current += 1

video.release()
cv2.destroyAllWindows()
# image_file = sys.argv[1]
