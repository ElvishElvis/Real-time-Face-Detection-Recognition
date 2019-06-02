import numpy as np
import cv2
import pandas as pd
import sys


def calculation(input):
    # print(input)
    # remember to call uploader before calculation, so that we have dataset in 'labeled_pics.csv file'!!!!!!!!!
    data = pd.read_csv('labeled_pics.csv', index_col=0)
    # Since we use the min number to define the best, so for empty we use max, so it will never be chosen
    results = data['features'].apply(
        lambda x: np.sqrt(np.sum(np.square(np.subtract(eval(x), input))) if x != '[]' else sys.float_info.max))
    print(list(results))
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
