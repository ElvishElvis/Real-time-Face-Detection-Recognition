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
    name_list=[]
    path_list=[]
    fic_list=[]
    infotable = pd.DataFrame(columns=['name', 'features', 'img'])
    with open(txtfile, encoding="utf16") as f:
        for line in f:
            if len(line)==0:
                continue
            name_picpath = line.split()
            #
            name = name_picpath[0]
            name_list.append(name)
            print("append "+name, end=" : ")
            path = name_picpath[1]
            path_list.append(path)
            print(path)
            print(" ")
        print("Uploading queue: \n")
        print(path_list)
        pics,fics,errors= calculate_feature(path_list)  # this function require us to input a path as parameter;
        errors=set(errors)
        print("Error index are {}".format(errors))
        valid_name=[ name_list[i] for i in range(len(name_list)) if i not in errors]
        invalid_name=[ name_list[i] for i in range(len(name_list)) if i in errors]
        for i in range(len(pics)):
            print(fics[i])
            new_table = pd.DataFrame(data={'name': [valid_name[i]], 'features': [fics[i]], 'img': [pics[i]]})
            infotable=pd.concat([infotable, new_table], ignore_index=True)
        print("\nSuccessful uploaded {} pics : ".format(len(pics)))
        print("Label names are : ")
        print(valid_name)
        print("\nFail to upload {} pics : ".format(len(errors)))
        print("Label names are : ")
        print(invalid_name)
    np.set_printoptions(threshold=sys.maxsize)

    infotable.to_csv('labeled_pics.csv', index=False)


def clear(csv_name):
    '''
    used for clearing the csv file
    :param csv_name: the name of the csv file that used for storing dataset
    '''

    f = open(csv_name, "w+")
    f.close()




clear("labeled_pics.csv")
uploader("tester.txt")
# print(calculation(np.random.random_sample((1,512))))

