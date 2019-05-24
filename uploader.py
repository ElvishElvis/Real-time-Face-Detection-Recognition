import pandas as pd
import cv2
import numpy as np
import re
import sys

from calculate import calculate_feature

from scipy import misc




def uploader(txtfile):


    infotable = pd.DataFrame(columns = ['name', 'features', 'img'])
    def upload_helper(string):

        name_picpath = string.split()
        name = name_picpath[0]

        picpath = name_picpath[1]

        img = cv2.imread(picpath)  # this is the picture in array form;

        img = misc.imresize(img, (160, 160), interp='bicubic')

        features = calculate_feature(picpath)[0] # this function require us to input a path as parameter;
        # print(features[0])
        
        new_table = pd.DataFrame(data = {'name': [name], 'features': [features], 'img': [img]})

        return pd.concat([infotable, new_table],ignore_index=True)



    with open(txtfile, encoding = "utf8") as f:
        for line in f:
            infotable = upload_helper(line)

    np.set_printoptions(threshold=sys.maxsize)

    infotable.to_csv('labeled_pics.csv', index = False)




def clear(csv_name):

    f = open(csv_name, "w+")
    f.close()


def calculation(input):
    # print(input)
    #remember to call uploader before calculation, so that we have dataset in 'labeled_pics.csv file'!!!!!!!!!
    data = pd.read_csv('labeled_pics.csv', index_col=0)

    # data['features'] = data['features'].apply(lambda x: re.sub('[\s\s]+', ' ', x.replace("\n ", '')).replace(' ', ','))
    
    # results=data['features'].apply(lambda x: np.sqrt(np.sum(np.square(np.subtract(x, input)))))
    #Since we use the min number to define the best, so for empty we use max, so it will never be chosen
    results = data['features'].apply(lambda x: np.sqrt(np.sum(np.square(np.subtract(eval(x), input))) if x != '[]' else sys.float_info.max))
    return results.idxmin()




uploader("tester.txt")
print(calculation(np.random.random_sample((1,512))))
#clear("labeled_pics.csv")










