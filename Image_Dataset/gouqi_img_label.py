# -*- coding: utf-8 -*-
"""
读取枸杞数据，将数据的存储路径和label写入csv文件
"""

##################### load packages #####################
import numpy as np
import os
from PIL import Image
from skimage import io, data
import scipy.io
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import cv2
import pandas as pd


labels = {'MQ2':0, 'NNQ9':1, 'NQ1':2, 'NQ5':3, 'NQ7':4, 'NQ8':5}


##################### load gouqi data ##########################
def gouqi_preprocess(gouqi_folder):
    '''
    gouqi_floder: gouqi original path 原始枸杞的路径
    '''

    ######## gouqi data path 数据保存路径 ########
    gouqi_dir = list()

    ######## gouqi data dirs 生成保存数据的名称 ########
    for img_dir in os.listdir(gouqi_folder):

        for img in os.listdir(os.path.join(gouqi_folder, img_dir)):
            gouqi_dir.append(img)

    ######## random shuffle gouqi data dirs 打乱数据的绝对路径和名称排序 ########
    random.shuffle(gouqi_dir)

    length = int(0.8 * len(gouqi_dir))


    ###################### gouqi train data #####################
    gouqi_train=[]
    
    for img_dir in gouqi_dir[0:length]:

        ######## get label ########
        label = labels[img_dir.split('_')[0]]
        
        gouqi_train.append(['./data/'+img_dir,label])

    gouqi_train_df = pd.DataFrame(gouqi_train)
    gouqi_train_df.to_csv("C:/Users/shaoqi/Desktop/gouqi_train.csv", index=False, header=None)
    
    print("train is over!")
    
    
    
    ###################### gouqi test data #####################
    gouqi_test=[]
    
    for img_dir in gouqi_dir[length:len(gouqi_dir)]:  

        ######## get label ########
        label = labels[img_dir.split('_')[0]]
        
        gouqi_test.append(['./data/'+img_dir,label])
    
    gouqi_test_df = pd.DataFrame(gouqi_test)
    gouqi_test_df.to_csv("C:/Users/shaoqi/Desktop/gouqi_test.csv", index=False, header=None)
    
    print("test is over!")


################ main函数入口 ##################
if __name__ == '__main__':
    ######### gouqi path 枸杞数据存放路径 ########
    gouqi_folder = 'D:\\SQ\\枸杞\\分类\\单个'

    ######## 数据预处理 ########
    gouqi_preprocess(gouqi_folder)