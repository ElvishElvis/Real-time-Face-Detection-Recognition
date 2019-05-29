import os
import pandas as pd
import cv2
import dlib
import numpy as np
import csv
from helpers import shape_to_np, resize

# get the datasets with emotions from our folder, and add them into a dictionary
data_dictionary = {}
for filename in os.listdir("Dataset"):
    pic_paths = []
    foldername = os.path.join('./Dataset/', filename)
    try:
        for pic_path in os.listdir(foldername):
            pic_path = os.path.join(foldername, pic_path)
            pic_paths.append(pic_path)
        data_dictionary[filename] = [pic_paths]
    except:
        pass

# transform the dictionary into a dataframe
emotion_pic_df = pd.DataFrame(data=data_dictionary, index=['pics']).T


# extract 68 features from those pictures
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def facial_pt_extractor(pic_path):
    ## Since we have the helper methods in helpers.py, we just need to import them
    # def rect_to_bb(rect):
    #     x = rect.left()
    #     y = rect.top()
    #     w = rect.right() - x
    #     h = rect.bottom() - y
    #     return (x, y, w, h)

    # def shape_to_np(shape, dtype="int"):
    #     coords = np.zeros((68, 2), dtype=dtype)
    #     for i in range(0, 68):
    #         coords[i] = (shape.part(i).x, shape.part(i).y)
    #
    #     return coords
    #
    # def resize(image, width=1200):
    #     r = width * 1.0 / image.shape[1]
    #     dim = (width, int(image.shape[0] * r))
    #     resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    #     return resized

    image = cv2.imread(pic_path)
    image = resize(image, width=1200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    coordinates = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        for (x, y) in shape:
            coordinates.append(x)
            coordinates.append(y)
    return coordinates


answer = emotion_pic_df['pics'].apply(lambda x: [facial_pt_extractor(i) for i in x])

def appending_list(original, adding):
    original_cp = original.copy()
    original_cp.append(adding)
    return original_cp


with_emotion = answer.to_frame().apply(lambda x: [appending_list(i, x.name) for i in x['pics']], axis = 1)


# write the data into csv file:
for row in with_emotion.sum():
    with open('emotions.csv', 'a',newline = '') as csvFile:
        file_is_empty = os.stat('emotions.csv').st_size == 0
        writer = csv.writer(csvFile)
        # if we don't have the file, we need to add an empty header into the file;
        if file_is_empty:
            writer.writerow([]*137)
        writer.writerow(row)
