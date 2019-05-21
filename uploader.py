import pandas as pd
import cv2
import numpy as np
import re
import sys

from calculate import calculate_feature

from scipy import misc

'''uploader
1. upload_helper(string)

if exist (csv): 已经存在一个csv dataset
    put our labeled pics into it.

    ---------------------------------------------------------------------
    name | extracted 512 feature array | picture in array form(1,160,160,3)
    --------------------------------------------------------------------

else: 没有一个存在的csv dataset
    create new csv dataset and put our labeled pics into it


2. uploader (txt)
for each line in txt:
    call upload_helper(each line)

3. clear (csv)

'''



def uploader(txtfile):
    '''
    the uploader for uploading our labeled pictures
    :param txtfile: txt file with rows of name and the path for our picture， each row is a different pic
    '''

    infotable = pd.DataFrame(columns = ['name', 'features', 'img'])
    def upload_helper(string, csv_name):
        '''
        the helper method for our uploader
        :param string: the string containing the name and path of our single picture
        :param csv_name: the csv file to put our dataset

        '''
        name_picpath = string.split()
        name = name_picpath[0]

        picpath = name_picpath[1]

        img = cv2.imread(picpath)  # this is the picture in array form;

        img = misc.imresize(img, (160, 160), interp='bicubic')

        features = calculate_feature(picpath) # this function require us to input a path as parameter;
        
        new_table = pd.DataFrame(data = {'name': [name], 'features': [features], 'img': [img]})

        return pd.concat([infotable, new_table],ignore_index=True)



    with open(txtfile, encoding = "utf16") as f:
        for line in f:
            infotable = upload_helper(line, 'labeled_pics.csv')

    np.set_printoptions(threshold=sys.maxsize)

    infotable.to_csv('labeled_pics.csv', index = False)




def clear(csv_name):
    '''
    used for clearing the csv file
    :param csv_name: the name of the csv file that used for storing dataset
    '''

    f = open(csv_name, "w+")
    f.close()


def calculation(input):
    #remember to call uploader before calculation, so that we have dataset in 'labeled_pics.csv file'!!!!!!!!!
    data = pd.read_csv('labeled_pics.csv', index_col=0)

    data['features'] = data['features'].apply(lambda x: re.sub('[\s\s]+', ' ', x.replace("\n ", '')).replace(' ', ','))

    results=data['features'].apply(lambda x: np.sqrt(np.sum(np.square(np.subtract(eval(x), input)))))
    return results.idxmin()




#uploader("tester.txt")
#print(calculation(np.random.random_sample((1,512))))
#clear("labeled_pics.csv")










