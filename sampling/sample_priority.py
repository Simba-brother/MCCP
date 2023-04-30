from cProfile import label
from turtle import color
from sympy import source
from tensorflow.keras.models import Model, Sequential, load_model
from PIL import Image
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import queue as Q # Q.PriorityQueue()
# from multiprocessing import Queue as Q
import math
import pickle
import joblib
import random
from sklearn.utils import shuffle
from collections import defaultdict
import tensorflow as tf
import heapq
import json
import re
from matplotlib import pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import re


os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def saveData(data, save_dir, save_fileName):

    # with open(filename, "wb+") as file_obj:
    #     pickle.dump(data, file_obj)
    save_path = os.path.join(save_dir, save_fileName)
    joblib.dump(data, save_path)

def deleteIgnoreFile(file_list):
    for item in file_list:
        if item.startswith('.'):# os.path.isfile(os.path.join(Dogs_dir, item)):
            file_list.remove(item)
    return file_list

def gen_df(dir_path, save_dir, save_file_name, source, dic_localLabel_to_globalLable):
    '''
    形成数据集的dataFrame
    args:
        dir_path: "../dataSets/overlap_datasets/weather/Weather_Image_Recognition/dataset_pure_split/val"
        save_dir: "./saved/weather/val"
        save_file_name: "Weather_Image_Recognition.csv"
    '''
    # 加载数据
    # dir_path = "../dataSets/overlap_datasets/sports/Sport_Classification_Dataset/data/dataset_pure_split/val"
    # dir_path = "../dataSets/overlap_datasets/weather/Weather_dataset/dataset_pure_split/val"
    # dir_path = "../dataSets/overlap_datasets/weather/Weather_Image_Recognition/dataset_pure_split/val"
    dir_name = os.listdir(dir_path)
    dir_name = deleteIgnoreFile(dir_name)
    dir_name.sort() # 很重要,这是坑！
    categories = dir_name
    filepaths=[]
    labels=[]
    labels_index = []
    source_list = []
    global_label_list = []
    for category in categories:
        class_path = os.path.join(dir_path, category)    
        file_names = os.listdir(class_path)
        file_names = deleteIgnoreFile(file_names)
        for file_name in file_names:
            file_path = os.path.join(class_path, file_name)
            filepaths.append(file_path)
            labels.append(category)
            labels_index.append(categories.index(category))
            global_label_list.append(dic_localLabel_to_globalLable.get(categories.index(category)))
            source_list.append(source)
    Fseries=pd.Series(filepaths, name='file_path')
    Lseries=pd.Series(labels, name='label')
    L_indexSeries = pd.Series(labels_index, name='local_label_index')
    source_Series = pd.Series(source_list, name='source')
    globalLabel_Series = pd.Series(global_label_list, name='global_label_index')
    Dataset_df = pd.concat([Fseries, Lseries, L_indexSeries, source_Series, globalLabel_Series], axis=1)
    save_path = os.path.join(save_dir, save_file_name)
    Dataset_df.to_csv(save_path, index=False)
    print("gen_df success!")

def concat(csv_1_filePath, csv_2_filePath, save_dir, save_fileName):
    '''
    将两个dataFrame 上下链接起来
    args:
        csv_1_filePath: "./saved/sports/val/73_Sports_Image_Classification.csv"
        csv_2_filePath: "./saved/sports/val/Sport_Classification_Dataset.csv"
        save_dir: "./saved/sports/val"
        save_fileName: "merged_df.csv"
    '''
    # 读取csv
    # df_1 = pd.read_csv("./saved/shapes/val/four_shapes_val_df.csv")
    # df_2 = pd.read_csv("./saved/shapes/val/Geometric_Shapes_Mathematics_df.csv")
    df_1 = pd.read_csv(csv_1_filePath)
    df_2 = pd.read_csv(csv_2_filePath)
    # df_1 = pd.read_csv("./saved/weather/val/Weather_dataset.csv")
    # df_2 = pd.read_csv("./saved/weather/val/Weather_Image_Recognition.csv")

    # print(four_shapes_val_df.head(5))
    # print(Geometric_Shapes_Mathematics_df.head(5))
    # print(four_shapes_val_df.shape)
    # four_shapes_val_df.rename(columns={'lables':'label'}, inplace=True)
    # Geometric_Shapes_Mathematics_df.rename(columns={'Unnamed: 0':'idx'})
    # print(four_shapes_val_df.columns)
    # print(Geometric_Shapes_Mathematics_df.columns)
    # print(four_shapes_val_df.head(5))
    merged_df = pd.concat([df_1, df_2], ignore_index=True)
    # merged_df.rename(columns={'Unnamed: 0':'idx'}, inplace=True)
    # print(merged_df["Unnamed: 0"])
    # print(merged_df.iloc[5992:])
    save_path = os.path.join(save_dir, save_fileName)
    merged_df.to_csv(save_path) # index=False
    # merged_df.to_csv('./saved/shapes/val/merged_df.csv', index=False)
    # merged_df.to_csv('./saved/weather/val/merged_df.csv', index=False)
    print("concat success")

def sameName():
    '''
    确保overlap 名字要一致，以第一个为准
    '''
    merged_df = pd.read_csv('./saved/sports/val/merged_df.csv') ## todo

    # merged_df.loc[merged_df['labels'] == 'fogsmog', 'labels'] = 'foggy'
    # merged_df.loc[merged_df['labels'] == 'rain', 'labels'] = 'rainy'

    merged_df.loc[merged_df['labels'] == 'formula1', 'labels'] = 'formula 1 racing'
    merged_df.loc[merged_df['labels'] == 'table_tennis', 'labels'] = 'table tennis'
    merged_df.loc[merged_df['labels'] == 'weight_lifting', 'labels'] = 'weightlifting'

    merged_df.to_csv('./saved/sports/val/merged_df.csv') #####!!!!
    print("sameName success")


def predic_ans(merged_csv_path, model_1, model_2, col_1_name, col_2_name, col_3_name, col_4_name,
    save_dir, save_fileName, target_size_1, target_size_2):
    '''
    看看各方模型在混合集上的预测label和confidence
    arg:
        merged_csv_path: './saved/sports/val/merged_df.csv'
        col_1_name: "73_Sports_Image_Classification_model_predict_labels"
        col_2_name: "73_Sports_Image_Classification_model_confidence"
        col_3_name: "Sport_Classification_Dataset_model_predict_labels"
        col_4_name: "Sport_Classification_Dataset_model_confidence"
        save_dir: './saved/sports/val'
        save_fileName: 'merged_df.csv'
        target_size_1: (150, 150) # Weather_dataset
        target_size_2: (224, 224)
    '''

    merged_df = pd.read_csv(merged_csv_path)
    ## Animal
    # datagen = ImageDataGenerator() 
    # datagen_2 = ImageDataGenerator()
    ## car_body_style(已经弃用)
    # datagen = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, rotation_range = 20) # rescale=1./255
    # datagen_2 = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, rotation_range = 20)
    ## custom_split
    datagen = ImageDataGenerator()
    datagen_2 = ImageDataGenerator()
    # 这里不需要关心classes
    batches = datagen.flow_from_dataframe(merged_df, x_col='file_path', y_col='label',
            target_size=target_size_1, 
            color_mode='rgb',  ## to change
            # classes=None, class_mode='sparse', 
            batch_size=32, 
            shuffle=False, 
            seed=123
            # save_to_dir=None, save_prefix='', save_format='png', subset=None, interpolation='nearest'
            )

    batches_2 = datagen_2.flow_from_dataframe(merged_df, x_col='file_path', y_col='label',
            target_size=target_size_2, 
            color_mode='rgb',
            # classes=None, class_mode='sparse',
            batch_size=32, 
            shuffle=False, 
            seed=123
            #save_to_dir=None, save_prefix='', save_format='png', subset=None, interpolation='nearest'
            )
    # images, labels = batches.next()
    # print(images[27].shape)
    # print(labels[27])
    # plt.imshow(images[27])
    # print("faf")
    # plt.show()
    # print("length:",len(batches))
    # 加载模型
    # model_1 = load_model("../dataSets/overlap_datasets/shapes/Four_shapes/saved/models/best-01-1.00.hdf5")
    # model_1 = load_model("../dataSets/overlap_datasets/sports/73_Sports_Image_Classification/saved/model/best_model_preTrain_73_Sports_Image_Classification.h5")
    # model_1 = load_model("../dataSets/overlap_datasets/weather/Weather_dataset/saved/model/best_vgg19_92.h5")

    # model_2 = load_model("../dataSets/overlap_datasets/shapes/Geometric_Shapes_Mathematics/six-shapes-dataset-v1/saved/model/my_EfficientNetB2-geometry-99.9.h5")
    # model_2 = load_model("../dataSets/overlap_datasets/sports/Sport_Classification_Dataset/saved/models/Sport_Classification_Dataset_ResNet50_89.h5")
    # model_2 = load_model("../dataSets/overlap_datasets/weather/Weather_Image_Recognition/saved/model/best_my_weather_93.08.h5")
    # output = model.predict(test_dataset_gen, batch_size = 32, verbose=1)

    # predict_array = model_1.predict(datagen, batch_size = 32, verbose=1)
    # predict_array_2 = model_2.predict(datagen_2, batch_size = 32, verbose=1)

    predict_array = model_1.predict_generator(batches, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    # # 获得伪标签
    # forecasts = np.argmax(predict_array, axis = 1)
    # # 根据groundTrue label 和 伪标签 获得混淆矩阵
    # cm = confusion_matrix(batches.classes, forecasts)
    # print(cm)
    # # 根据groundTrue label 和 伪标签 获得分类报告
    # print(classification_report(batches.classes, forecasts))
    # # 获得batchs的总体acc
    # acc = model_1.evaluate(batches, batch_size = 32, verbose=1, return_dict=True)
    # print(acc)

    predict_array_2 = model_2.predict_generator(batches_2, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    # predict_array = model_1(batches)
    # predict_array_2 = model_2(batches)
    # print(predict_array.shape)
    # print(predict_array)
    predict_labels = []
    predict_labels_2 = []
    confidence_list = []
    probability_list_1 = []
    confidence_list_2 = []
    probability_list_2 = []
    for item in predict_array:
        predict_labels.append(np.argmax(item))
        confidence_list.append(np.max(item))
        probability_list_1.append(item)
    for item2 in predict_array_2:
        predict_labels_2.append(np.argmax(item2))
        confidence_list_2.append(np.max(item2))
        probability_list_2.append(item2)
    # predict_labels_series=pd.Series(predict_labels, name='four_model_predict_labels')
    # confidence_list_series=pd.Series(confidence_list, name='four_model_confidence')
    # predict_labels_series_2=pd.Series(predict_labels_2, name='Geometric_Shapes_Mathematics_model_predict_labels')
    # confidence_list_series_2=pd.Series(confidence_list_2, name='Geometric_Shapes_Mathematics_model_confidence')
    predict_labels_series=pd.Series(predict_labels, name=col_1_name)
    confidence_list_series=pd.Series(confidence_list, name=col_2_name)
    probability_list_1_series = pd.Series(probability_list_1, name="probabilty_partyOne_list")
    predict_labels_series_2=pd.Series(predict_labels_2, name=col_3_name)
    confidence_list_series_2=pd.Series(confidence_list_2, name=col_4_name)
    probability_list_2_series  = pd.Series(probability_list_2, name="probabilty_partyTwo_list")
    # predict_labels_series=pd.Series(predict_labels, name='Weather_dataset_model_predict_labels')
    # confidence_list_series=pd.Series(confidence_list, name='Weather_dataset_model_confidence')
    # predict_labels_series_2=pd.Series(predict_labels_2, name='Weather_Image_Recognition_model_predict_labels')
    # confidence_list_series_2=pd.Series(confidence_list_2, name='Weather_Image_Recognition_model_confidence')

    merged_df = pd.concat([merged_df, predict_labels_series, predict_labels_series_2,  confidence_list_series, confidence_list_series_2, probability_list_1_series, probability_list_2_series], axis=1)
    # merged_df = pd.concat([merged_df, probability_list_1_series, probability_list_2_series], axis=1)
    save_path = os.path.join(save_dir, save_fileName)
    merged_df.to_csv(save_path, index=False)
    print("predic_ans() success")

def combination_model_predict(merged_csv_path, 
                            combination_model, 
                            col_1_name,
                            col_2_name,
                            save_dir, save_fileName, target_size_1, target_size_2):
    '''
    为我们的merged_csv补充combination_model的信息
    '''
    # 加载待预测数据集
    merged_df = pd.read_csv(merged_csv_path)
    num = merged_df.shape[0]
    # 数据批次生成器
    generator_left = ImageDataGenerator()
    generator_right = ImageDataGenerator()
    batch_size = 32
    classes = merged_df["label"].unique()
    # 全局类别,字典序列
    classes = np.sort(classes).tolist()
    # 获得批次数据
    batches = generate_generator_multiple(generator_left, generator_right, merged_df, batch_size, classes, target_size_1, target_size_2)
    # 获得预测数据
    predict_array = combination_model.predict(batches, batch_size=batch_size, verbose=1, steps=num/batch_size)
    # 收集每个样本的预测标签
    predict_label_list = []
    # 收集每个样本Top_confidence
    confidence_list = []
    # 收集每个样本预测置信度列表
    probability_list = []
    for item in predict_array:
        predict_label_list.append(np.argmax(item))
        confidence_list.append(np.max(item))
        probability_list.append(item)
    predict_label_series=pd.Series(predict_label_list, name=col_1_name)
    confidence_series = pd.Series(confidence_list, name = col_2_name)
    probability_series = pd.Series(probability_list, name = "probabilty_list")
    # 收集到的列信息
    merged_df = pd.concat([merged_df, predict_label_series,  confidence_series, probability_series], axis=1)
    save_path = os.path.join(save_dir, save_fileName)
    merged_df.to_csv(save_path, index=False)

def predic_trueOrFalse(merged_csv_path, model_1_localLabel_to_globalLable, model_2_localLabel_to_globalLable, save_dir, save_fileName):
    '''
    model_1_localLabel_to_globalLable: {0:0, 1:1, 2:5, 3:7, 4:9}
    model_2_localLabel_to_globalLable: {0:1, 1:2, 2:3, 3:4, 4:6, 5:8}
    '''
    merged_df = pd.read_csv(merged_csv_path)
    model_1_predic_isTrue = []
    model_2_predic_isTrue = []
    for index, row in merged_df.iterrows():
        global_label_index = row["global_label_index"]
        model_1_globalLabel_index = model_1_localLabel_to_globalLable[row["model_1_predict_label"]]
        model_2_globalLabel_index = model_2_localLabel_to_globalLable[row["model_2_predict_label"]]
        if model_1_globalLabel_index == global_label_index:
            model_1_predic_isTrue.append(1)
        else:
            model_1_predic_isTrue.append(0)
        if model_2_globalLabel_index == global_label_index:
            model_2_predic_isTrue.append(1)
        else:
            model_2_predic_isTrue.append(0)
    model_1_predic_isTrue_Series = pd.Series(model_1_predic_isTrue, name="model_1_predic_isTrue")
    model_2_predic_isTrue_Series=pd.Series(model_2_predic_isTrue, name="model_2_predic_isTrue")
    merged_df = pd.concat([merged_df, model_1_predic_isTrue_Series, model_2_predic_isTrue_Series], axis=1)
    save_path = os.path.join(save_dir, save_fileName)
    merged_df.to_csv(save_path, index=False)
    print("predic_trueOrFalse() success")

def is_overlap(dir_path, dir_path_overlap, csv_path, col_name, extend_col_name, save_dir, save_fileName):
    '''
    判断伪标签, 是否是overlap
    args:
        dir_path: "../dataSets/overlap_datasets/sports/Sport_Classification_Dataset/data/dataset_pure_split/val"
        dir_path_overlap: "../dataSets/overlap_datasets/sports/overlap_dataset/Sport_Classification_Dataset"
        csv_path: './saved/sports/val/merged_df.csv'
        col_name: 'Sport_Classification_Dataset_model_predict_labels'
        extend_col_name: 'Sport_Classification_Dataset_model_predict_isOverlap'
        save_dir: 
        save_fileName
    '''
    # dir_path = "../dataSets/overlap_datasets/shapes/Four_shapes/dataset/val"
    # dir_path = "../dataSets/overlap_datasets/shapes/Geometric_Shapes_Mathematics/six-shapes-dataset-v1/dataset_pure_split/val"
    # dir_path = "../dataSets/overlap_datasets/sports/73_Sports_Image_Classification/dataset_pure_split/val"
    # dir_path = "../dataSets/overlap_datasets/weather/Weather_dataset/dataset_pure_split/val"
    # dir_path = "../dataSets/overlap_datasets/weather/Weather_Image_Recognition/dataset_pure_split/val"
    dir_name = os.listdir(dir_path)
    dir_name = deleteIgnoreFile(dir_name)
    dir_name.sort()

    # dir_path_overlap = "../dataSets/overlap_datasets/shapes/overlap_dataset/Four_shapes"
    # dir_path_overlap = "../dataSets/overlap_datasets/shapes/overlap_dataset/Geometric_Shapes_Mathematics"
    # dir_path_overlap = "../dataSets/overlap_datasets/sports/overlap_dataset/73_Sports_Image_Classification"

    # dir_path_overlap = "../dataSets/overlap_datasets/weather/overlap_dataset/Weather_Image_Recognition"
    dir_name_overlap = os.listdir(dir_path_overlap)
    dir_name_overlap = deleteIgnoreFile(dir_name_overlap)
    dir_name_overlap.sort()
     
    overlap_classes = []
    for category in dir_name_overlap:
        overlap_classes.append(dir_name.index(category))
    merged_df = pd.read_csv(csv_path)
    is_overlap = []
    # for pre_label in merged_df['four_model_predict_labels']:
    # for pre_label in merged_df['Geometric_model_predict_labels']:
    # for pre_label in merged_df['73_Sports_model_predict_labels']:
    for pre_label in merged_df[col_name]:
        if pre_label in overlap_classes:
            is_overlap.append(1)
        else:
            is_overlap.append(0)    
    # predic_isOverlap_serices = pd.Series(is_overlap, name="four_model_predic_isOverlap")
    # predic_isOverlap_serices = pd.Series(is_overlap, name="geometric_model_predic_isOverlap")
    # predic_isOverlap_serices = pd.Series(is_overlap, name="73_Sports_model_predic_isOverlap")
    predic_isOverlap_serices = pd.Series(is_overlap, name=extend_col_name)
    merged_df = pd.concat([merged_df, predic_isOverlap_serices], axis=1)
    save_path = os.path.join(save_dir, save_fileName)
    merged_df.to_csv(save_path, index=False)
    print("is_overlap success")

def isConsistence(csv_path, 
    dir_path_1, dir_path_overlap_1,
    dir_path_2, dir_path_overlap_2,
    model_1_predict_label, model_2_predict_label,
    extend_col_name,save_dir, save_fileName):
    '''
    判断两方model预测的结果是否一致
    args
        csv_path: './saved/sports/val/merged_df.csv'
        dir_path_1: "../dataSets/overlap_datasets/sports/73_Sports_Image_Classification/dataset_pure_split/val"
        dir_path_overlap_1: "../dataSets/overlap_datasets/sports/overlap_dataset/73_Sports_Image_Classification"
        dir_path_2: "../dataSets/overlap_datasets/sports/Sport_Classification_Dataset/data/dataset_pure_split/val"
        dir_path_overlap_2: "../dataSets/overlap_datasets/sports/overlap_dataset/Sport_Classification_Dataset"
        extend_col_name: "isConsistence"
        save_dir:
        save_fileName

    '''
    # 读取df
    merged_df = pd.read_csv(csv_path)
    # 获得各方的全类和overlap类
    def help(dir_path, dir_path_overlap):
        # dir_path = "../dataSets/overlap_datasets/shapes/Geometric_Shapes_Mathematics/six-shapes-dataset-v1/dataset_pure_split/val"
        dir_name = os.listdir(dir_path)
        dir_name = deleteIgnoreFile(dir_name)
        dir_name.sort()

        # dir_path_overlap = "../dataSets/overlap_datasets/shapes/overlap_dataset/Geometric_Shapes_Mathematics"
        dir_name_overlap = os.listdir(dir_path_overlap)
        dir_name_overlap = deleteIgnoreFile(dir_name_overlap)
        dir_name_overlap.sort()
        return dir_name, dir_name_overlap
    # dir_name_1, dir_name_overlap_1 = help("../dataSets/overlap_datasets/shapes/Four_shapes/dataset/val", "../dataSets/overlap_datasets/shapes/overlap_dataset/Four_shapes")
    # dir_name_2, dir_name_overlap_2 = help("../dataSets/overlap_datasets/shapes/Geometric_Shapes_Mathematics/six-shapes-dataset-v1/dataset_pure_split/val", "../dataSets/overlap_datasets/shapes/overlap_dataset/Geometric_Shapes_Mathematics")
    dir_name_1, dir_name_overlap_1 = help(dir_path_1, dir_path_overlap_1)
    dir_name_2, dir_name_overlap_2 = help(dir_path_2, dir_path_overlap_2)
    # dir_name_1, dir_name_overlap_1 = help("../dataSets/overlap_datasets/weather/Weather_dataset/dataset_pure_split/val", "../dataSets/overlap_datasets/weather/overlap_dataset/Weather_dataset")
    # dir_name_2, dir_name_overlap_2 = help("../dataSets/overlap_datasets/weather/Weather_Image_Recognition/dataset_pure_split/val", "../dataSets/overlap_datasets/weather/overlap_dataset/Weather_Image_Recognition")
    # 获得overlap对应class_idx
    list_class_idx_1 = []
    list_class_idx_2 = []
    for category in dir_name_overlap_1:
        list_class_idx_1.append(dir_name_1.index(category))
    for category_2 in dir_name_overlap_2:
        list_class_idx_2.append(dir_name_2.index(category_2))
    # 存储对应关系
    dic_1_to_2 = {}
    dic_2_to_1 = {}
    for class_idx_1, class_idx_2 in zip(list_class_idx_1, list_class_idx_2):
        dic_1_to_2[class_idx_1] = class_idx_2
        dic_2_to_1[class_idx_2] = class_idx_1
    
    isConsistence = []
    for index, row in merged_df.iterrows():
        # one_model_predict_label = row["four_model_predict_labels"]
        # two_model_predict_label = row["Geometric_Shapes_Mathematics_model_predict_labels"]
        one_model_predict_label = row[model_1_predict_label]
        two_model_predict_label = row[model_2_predict_label]
        # one_model_predict_label = row["Weather_dataset_model_predict_labels"]
        # two_model_predict_label = row["Weather_Image_Recognition_model_predict_labels"]
        if one_model_predict_label in dic_1_to_2.keys() and dic_1_to_2[one_model_predict_label] == two_model_predict_label:
            isConsistence.append(1)
        else:
            isConsistence.append(0)
    isConsistence_series = pd.Series(isConsistence, name=extend_col_name)
    merged_df = pd.concat([merged_df, isConsistence_series], axis=1)
    save_path = os.path.join(save_dir, save_fileName)
    merged_df.to_csv(save_path, index=False)
    print("isConsistence success")

def global_label_index(csv_path, part_A_labelToGlobal, part_B_labelToGlobal, save_dir, save_fileName):
    merged_df = pd.read_csv(csv_path)
    global_label_list = []
    for index, row in merged_df.iterrows():
        source = row["source"]
        labels_index = row["labels_index"]
        if source == 1:
            global_label = part_A_labelToGlobal[labels_index]
            global_label_list.append(global_label)
        if source == 2:
            global_label = part_B_labelToGlobal[labels_index]
            global_label_list.append(global_label)
    global_label = pd.Series(global_label_list, name="global_label")
    merged_df = pd.concat([merged_df, global_label], axis=1)
    save_path = os.path.join(save_dir, save_fileName)
    merged_df.to_csv(save_path, index=False)
    print("global_label_index() success")

def shuffle_dataFrame(csv_path, save_dir, file_name):
    merged_df = pd.read_csv(csv_path)
    merged_df = shuffle(merged_df, random_state=123)
    save_path = os.path.join(save_dir, file_name)
    merged_df.to_csv(save_path, index=False)
    print("shuffle_dataFrame() success")

def getQueue(csv_path,
    model_1_confidence, model_2_confidence,
    model_1_predict_label_isOverlap,model_2_predict_label_isOverlap,
    isConsis):
    '''
    args:
        csv_path: './saved/'+scene+'/val/merged_df.csv'
        model_1_confidence: "four_model_confidence"
        model_2_confidence: "Geometric_Shapes_Mathematics_model_confidence"
        model_1_predict_label_isOverlap: "four_model_predic_isOverlap"
        model_2_predict_label_isOverlap: "Geometric_Shapes_Mathematics_model_predict_isOverlap"
        isConsis:"isConsistence"
    '''
    queue_consistence_overlap = Q.PriorityQueue()
    queue_consistence_unOverlap = Q.PriorityQueue()
    queue_unConsistence_allOverlap = Q.PriorityQueue()
    queue_unConsistence_existOverlap = Q.PriorityQueue()
    queue_unConsistence_allUnique = Q.PriorityQueue()
    merged_df = pd.read_csv(csv_path)
    # print(merged_df.head(5))
    for index, row in merged_df.iterrows():
        # print(index, row)

        one_model_confidence = row[model_1_confidence]
        two_model_confidence = row[model_2_confidence]
        one_model_predic_isOverlap = row[model_1_predict_label_isOverlap]
        two_model_predic_isOverlap = row[model_2_predict_label_isOverlap]

        # one_model_confidence = row["73_Sports_Image_Classification_model_confidence"]
        # two_model_confidence = row["Sport_Classification_Dataset_model_confidence"]
        # one_model_predic_isOverlap = row["73_Sports_Image_Classification_model_predict_isOverlap"]
        # two_model_predic_isOverlap = row["Sport_Classification_Dataset_model_predict_isOverlap"]

        # one_model_confidence = row["Weather_dataset_model_confidence"]
        # two_model_confidence = row["Weather_Image_Recognition_model_confidence"]
        # one_model_predic_isOverlap = row["Weather_dataset_model_predic_isOverlap"]
        # two_model_predic_isOverlap = row["Weather_Image_Recognition_model_predic_isOverlap"]


        isConsistence = row[isConsis]
        if isConsistence == 1 and one_model_predic_isOverlap == 1 and two_model_predic_isOverlap == 1:
            temp = []
            dif = -abs(one_model_confidence - two_model_confidence)  # dif越大优先级越高
            temp.append(dif)
            temp.append(index)
            queue_consistence_overlap.put(temp)
        elif isConsistence == 1 and one_model_predic_isOverlap == 0:   
            temp = []
            temp.append(one_model_confidence)
            temp.append(index)
            queue_consistence_unOverlap.put(temp)
        elif isConsistence == 1 and two_model_predic_isOverlap == 0: 
            temp = []
            temp.append(two_model_confidence)
            temp.append(index)
            queue_consistence_unOverlap.put(temp)
        elif isConsistence == 0 and one_model_predic_isOverlap == 1 and two_model_predic_isOverlap == 1:
            # 预测结果不一致，但是都在overlap 这个权重挺大
            temp = []
            dif = abs(one_model_confidence - two_model_confidence)
            temp.append(dif)
            temp.append(index)
            queue_unConsistence_allOverlap.put(temp)
        elif isConsistence == 0 and one_model_predic_isOverlap == 1 or two_model_predic_isOverlap == 1:
            # 预测结果不一致，但是存在一个overlap
            temp = []
            dif = abs(one_model_confidence - two_model_confidence)
            temp.append(dif)
            temp.append(index)
            queue_unConsistence_existOverlap.put(temp)
        elif isConsistence == 0 and one_model_predic_isOverlap == 0 and two_model_predic_isOverlap == 0:
            # 预测结果不一致，都认为自己unique
            temp = []
            dif = abs(one_model_confidence - two_model_confidence)
            temp.append(dif)
            temp.append(index)
            queue_unConsistence_allUnique.put(temp)
    print("getQueue success")
    return queue_consistence_overlap, queue_consistence_unOverlap, queue_unConsistence_allOverlap, queue_unConsistence_existOverlap, queue_unConsistence_allUnique

def getQueue_2(csv_path,
                model_1_confidence, model_2_confidence,
                isConsis):
    '''
    得到一个更可能是对抗样本队列 和 一个更可能是非overlap队列
    args:
        csv_path: './saved/'+scene+'/val/merged_df.csv'
        model_1_confidence: "four_model_confidence"
        model_2_confidence: "Geometric_Shapes_Mathematics_model_confidence"
    '''
    queue_not_consistence = Q.PriorityQueue()
    queue_consistence = Q.PriorityQueue()
    merged_df = pd.read_csv(csv_path)
    for index, row in merged_df.iterrows():
        one_model_confidence = row[model_1_confidence]
        two_model_confidence = row[model_2_confidence]
        isConsistence = row[isConsis]
        dif = abs(one_model_confidence - two_model_confidence)
        if isConsistence == 0:
            # 如果伪标签不一致
            cur_dif = -dif  # dif 越高 优先级越高 即 更可能是对抗样本, 同时 队列的尾端更可能是异常（即非overlap),可谓是一举两得
            temp = []
            temp.append(cur_dif)
            temp.append(index)
            queue_not_consistence.put(temp)

        elif isConsistence == 1:
            # 如果伪标签一致
            cur_dif = -dif  # dif 越高 优先级越高 即 两人意见统一，但不是那么统一， 这个队列几乎是ovlap中很共识的样本，没有reTrain意义
            temp = []
            temp.append(cur_dif)
            temp.append(index)
            queue_consistence.put(temp)
    return queue_not_consistence, queue_consistence

def getQueue_3(csv_path):
    # 双方都预测一致
    queue_consistence = Q.PriorityQueue()
    # 是一方的误分类
    queue_adversary = Q.PriorityQueue()
    # 分不太轻 是，一方的误分类 还是 一方的异常。但是，队头更可能是误分类，队尾更可能是 异常
    queue_adversary_exception = Q.PriorityQueue()
    # 是一方的异常
    queue_exception = Q.PriorityQueue()
    merged_df = pd.read_csv(csv_path)
    for index, row in merged_df.iterrows():
        one_model_confidence = row["model_1_confidence"]
        two_model_confidence = row["model_2_confidence"]
        one_model_predic_isOverlap = row['model_1_predict_label_isOverlap']
        two_model_predic_isOverlap = row['model_2_predict_label_isOverlap']
        isConsist = row["isConsistence"]
        dif = abs(one_model_confidence - two_model_confidence)
        if isConsist == 1:
            cur_dif = -dif
            temp = []
            temp.append(cur_dif)
            temp.append(index)
            queue_consistence.put(temp)
        elif one_model_predic_isOverlap == 1 and two_model_predic_isOverlap == 1:
            cur_dif = -dif
            queue_adversary.put([cur_dif, index])
        elif one_model_predic_isOverlap == 1 or two_model_predic_isOverlap == 1:
            cur_dif = -dif
            queue_adversary_exception.put([cur_dif, index])
        elif one_model_predic_isOverlap == 0 or two_model_predic_isOverlap == 0:
            cur_dif = -dif
            queue_exception.put([cur_dif, index])
    return queue_consistence, queue_adversary, queue_exception ,queue_adversary_exception

def getQueue_4(csv_path, label_space, label_AToGlobal, label_BToGlobal):
    '''
    labelOneToLableTwo: label空间对应关系（字典类型）
    label_AToGlobal: label A 对应 到 全局 是几
    label_BToGlobal: label B 对应 到 全局 是几
    '''   
    merged_df = pd.read_csv(csv_path)  
    queue_list = []
    for i in range(len(label_space)):
        queue_list.append(Q.PriorityQueue())
    queue_ambiguous = Q.PriorityQueue()
    for index, row in merged_df.iterrows():
        one_model_confidence = row["model_1_confidence"]
        one_model_confidence = round(one_model_confidence, 2)
        two_model_confidence = row["model_2_confidence"]
        two_model_confidence = round(two_model_confidence, 2)
        isConsist = row["isConsistence"]
        model_1_predict_label = row["model_1_predict_label"]
        model_2_predict_label = row["model_2_predict_label"]
        dif = abs(one_model_confidence - two_model_confidence)
        if isConsist == 1:
            cur_dif = -dif
            global_label_pre = label_AToGlobal[model_1_predict_label]
            queue = queue_list[global_label_pre]
            queue.put([cur_dif, index])
        elif one_model_confidence > two_model_confidence:
            cur_dif = -dif
            global_label_pre = label_AToGlobal[model_1_predict_label]
            queue = queue_list[global_label_pre]
            queue.put([cur_dif, index])
        elif two_model_confidence > one_model_confidence:
            cur_dif = -dif
            global_label_pre = label_BToGlobal[model_2_predict_label]
            queue = queue_list[global_label_pre]
            queue.put([cur_dif, index])
        elif one_model_confidence == two_model_confidence:
            queue_ambiguous.put([0, index]) # 模棱两可
    return queue_list, queue_ambiguous

def get_Queue5(csv_path):
    queue_agree  = Q.PriorityQueue()
    queue_adversary = Q.PriorityQueue()
    queue_adv_A_ex_B = Q.PriorityQueue()
    queue_adv_B_ex_A = Q.PriorityQueue()
    queue_exception = Q.PriorityQueue()
    merged_df = pd.read_csv(csv_path) 
    for index, row in merged_df.iterrows():  ## 注意这个index 是 从0开始的，就是这个dataFrame的单纯的row_index
        one_model_confidence = row["model_1_confidence"]
        one_model_confidence = round(one_model_confidence, 2)
        two_model_confidence = row["model_2_confidence"]
        two_model_confidence = round(two_model_confidence, 2) 
        isConsist = row["isConsistence"] 
        dif = abs(one_model_confidence - two_model_confidence)
        model_1_predict_label_isOverlap =  row["model_1_predict_label_isOverlap"]
        model_2_predict_label_isOverlap = row["model_2_predict_label_isOverlap"]
        if isConsist == 1:
            cur_dif = -dif 
            queue_agree.put([cur_dif, index])
        elif model_1_predict_label_isOverlap == 1 and model_2_predict_label_isOverlap == 1:
            cur_dif = -dif
            queue_adversary.put([cur_dif, index])
        elif model_1_predict_label_isOverlap == 0 and model_2_predict_label_isOverlap == 1:
            cur_dif = -dif
            queue_adv_A_ex_B.put([cur_dif, index])
        elif model_1_predict_label_isOverlap == 1 and model_2_predict_label_isOverlap == 0:
            cur_dif = -dif
            queue_adv_B_ex_A.put([cur_dif, index])
        elif model_1_predict_label_isOverlap == 0 and model_2_predict_label_isOverlap == 0:
            cur_dif = dif
            queue_exception.put([cur_dif, index])
    return queue_agree, queue_adversary, queue_adv_A_ex_B, queue_adv_B_ex_A, queue_exception


def str_probabilty_list_To_list(str_data):
    '''
    "[0.3 0.4 0.2 0.1]" => [0.3, 0.4, 0.2, 0.1]:list
    '''
    ans_list = []
    data_list = re.split("\s+",str_data)
    data_list[0] = data_list[0].split("[")[1]
    data_list[-1] = data_list[-1].split("]")[0]
    for data in data_list:
        if data == '':
            continue
        ans_list.append(np.float32(data))
    return ans_list

def caculateDeepGini(list_data):
    '''
    deep_gini计算公式, deep_gini 越大 说明模型对此样例越不自信。
    '''
    deep_gini = 0
    var = 0
    for item in list_data:
        var += item*item
    deep_gini = 1 - var
    return deep_gini

def get_DeepGiniQueue(csv_path, measure, algorithm_flag):
    '''
    DeepGini(ISSTA'20) 的 priorityQueue
    args:
        measure: max | sum
        algorithm_flag: isolate | combination
    '''
    merged_df = pd.read_csv(csv_path) 
    queue = Q.PriorityQueue()
    if algorithm_flag == "isolate":
        for index, row in merged_df.iterrows():
            str_probabilty_partyOne = row["probabilty_partyOne_list"]
            str_probabilty_partyTwo = row["probabilty_partyTwo_list"]
            probabilty_partyOne_list = str_probabilty_list_To_list(str_probabilty_partyOne)
            probabilty_partyTwo_list = str_probabilty_list_To_list(str_probabilty_partyTwo)
            deepGini_partyOne = caculateDeepGini(probabilty_partyOne_list)
            deepGini_partyTwo = caculateDeepGini(probabilty_partyTwo_list)
            if measure == "sum":
            # 这个地方其实可以换度量方式
                deepGini_total = deepGini_partyOne + deepGini_partyTwo
            elif measure == "max":
                deepGini_total = max(deepGini_partyOne, deepGini_partyTwo)
            # deepGini越大优先级越高
            deepGini_total = -deepGini_total
            queue.put([-deepGini_total, index])
    elif algorithm_flag == "combination":
        # 遍历数据集
        for index, row in merged_df.iterrows():
            str_probabilties = row["probabilty_list"]
            probabilty_list = str_probabilty_list_To_list(str_probabilties)
            deepGini = caculateDeepGini(probabilty_list)
            # deepGini越大优先级越高
            queue.put([-deepGini, index])
    else:
        raise Exception("algorithm_flag出错")
    return queue

def get_MCPQueue(csv_path, source, algorithm_flag):
    '''
    MCP(ASC'20) 的priorityQueue
    return:
        queue_matrix: row_index:top_1_index, col_index:top_2_index
    '''
    merged_df = pd.read_csv(csv_path) 
    if algorithm_flag == "isolate":
        part_df = merged_df.loc[merged_df['source'] == source]
        local_classes = part_df["label"].unique()
        # 局部类别
        local_classes = np.sort(local_classes).tolist()
        # 构造出队列矩阵
        queue_matrix = [[Q.PriorityQueue() for j in range(len(local_classes))] for i in range(len(local_classes))]
        for index, row in merged_df.iterrows(): 
            probabilty_party_list = []
            if source == 1:
                probabilty_party_list = str_probabilty_list_To_list(row["probabilty_partyOne_list"])
            elif source == 2:
                probabilty_party_list = str_probabilty_list_To_list(row["probabilty_partyTwo_list"])
            else:
                raise Exception("source 没传对")
            # 获得top_2的索引
            localLabel_index_list = heapq.nlargest(2, range(len(probabilty_party_list)), probabilty_party_list.__getitem__)
            top_1_index = localLabel_index_list[0]
            top_2_index = localLabel_index_list[1]
            maxP = probabilty_party_list[top_1_index]
            secondP = probabilty_party_list[top_2_index]
            priority = maxP / secondP
            cur_queue = queue_matrix[top_1_index][top_2_index]
            cur_queue.put([priority, index])
        return queue_matrix
    elif algorithm_flag == "combination":
        global_classes = merged_df["label"].unique()    
        global_classes = np.sort(global_classes).tolist()
        # 构造出队列矩阵
        queue_matrix = [[Q.PriorityQueue() for j in range(len(global_classes))] for i in range(len(global_classes))]
        for index, row in merged_df.iterrows(): 
            probabilty_list = str_probabilty_list_To_list(row["probabilty_list"])
            # 获得top_2的索引
            globalLabel_index_list = heapq.nlargest(2, range(len(probabilty_list)), probabilty_list.__getitem__)
            top_1_index = globalLabel_index_list[0]
            top_2_index = globalLabel_index_list[1]
            maxP = probabilty_list[top_1_index]
            secondP = probabilty_list[top_2_index]
            priority = maxP / secondP
            cur_queue = queue_matrix[top_1_index][top_2_index]
            cur_queue.put([priority, index])
        return queue_matrix
    else:
        raise Exception("algorithm_flag 错误")

def probabilityStrTolist(probality_str):
    '''
    csv中的概率串To概率list
    '''
    ans_list = []
    data_list = re.split("\s+",probality_str)
    data_list[0] = data_list[0].split("[")[1]
    data_list[-1] = data_list[-1].split("]")[0]
    for data in data_list:
        if data == '':
            continue
        ans_list.append(np.float32(data))
    return ans_list


def priority_to_MCP_priority(my_queue, merged_df):
    queue_mcp = Q.PriorityQueue()
    while not my_queue.empty():
        priority, row_idx = my_queue.get()
        cur_row = merged_df.iloc[row_idx,:]
        probabilty_list = probabilityStrTolist(cur_row["probabilty_list"]) 
        globalLabel_TopIndex_list = heapq.nlargest(2, range(len(probabilty_list)), probabilty_list.__getitem__)
        top_1_index = globalLabel_TopIndex_list[0]
        top_2_index = globalLabel_TopIndex_list[1]
        maxP = probabilty_list[top_1_index]
        secondP = probabilty_list[top_2_index]
        mcp_priority = maxP / secondP
        queue_mcp.put([mcp_priority, row_idx])
    return queue_mcp

def group_mcp(merged_df, queue_agree, queue_adv_A_exp_B, queue_adv_B_exp_A, queue_exp):
    '''
    把这些队列里面的数据按照mcp优先级重新排列一下
    '''
    queue_agree_mcp = priority_to_MCP_priority(queue_agree, merged_df)
    queue_adv_A_exp_B_mcp = priority_to_MCP_priority(queue_adv_A_exp_B, merged_df)
    queue_adv_B_exp_A_mcp = priority_to_MCP_priority(queue_adv_B_exp_A, merged_df)
    queue_exp_mcp = priority_to_MCP_priority(queue_exp, merged_df)
    return queue_agree_mcp, queue_adv_A_exp_B_mcp, queue_adv_B_exp_A_mcp, queue_exp_mcp

def statistic_unique_Label(csv_path, unique_label_list_A, unique_label_list_B):
    '''
    统计unique label confidence 情况
    args:
        csv_path = "./saved/real/animal/merge/val/merged_predic_trueOrFalse_isOverlap_again_isConsis_isOverlapGT_shuffle_df.csv"
        unique_label_list_A = ["cat", "dog", "horse", "lion"]
        unique_label_list_B = ["butterfly", "chicken", "spider"]
    return:
        ans = {
            "cat":[n0, n1, n2, n3, n4, n5]  
            。。。。
        }
        # n0: predic is true; C自己方 > C对方 
        # n1: predic is true; C自己方 < C对方 
        # n2: predic is true; C自己方 == C对方 
        # n3: predic is false; C自己方 > C对方 
        # n4: predic is false; C自己方 < C对方 
        # n5: predic is false; C自己方 == C对方 
    '''
    merged_df = pd.read_csv(csv_path) 
    ans = {}
    for index, row in merged_df.iterrows():
        model_1_confidence = row["model_1_confidence"]
        model_2_confidence = row["model_2_confidence"]
        model_1_predic_isTrue = row["model_1_predic_isTrue"]
        model_2_predic_isTrue = row["model_2_predic_isTrue"]
        label = row["label"]
        if label in unique_label_list_A:
            ##  此样本是A unique
            if label not in ans.keys():
                ans[label] = [0 for i in range(6)]
            cur_list = ans.get(label)
            if model_1_predic_isTrue == 1:
                if model_1_confidence > model_2_confidence:
                    cur_list[0] += 1
                elif model_1_confidence < model_2_confidence:
                    cur_list[1] += 1
                else:
                    cur_list[2] += 1
            else:
                if model_1_confidence > model_2_confidence:
                    cur_list[3] += 1
                elif model_1_confidence < model_2_confidence:
                    cur_list[4] += 1
                else:
                    cur_list[5] += 1
        elif label in unique_label_list_B:
            ##  此样本是B unique
            if label not in ans.keys():
                ans[label] = [0 for i in range(6)]
            cur_list = ans.get(label)
            if model_2_predic_isTrue == 1:
                if model_2_confidence > model_1_confidence:
                    cur_list[0] += 1
                elif model_2_confidence < model_1_confidence:
                    cur_list[1] += 1
                else:
                    cur_list[2] += 1
            else:
                if model_2_confidence > model_1_confidence:
                    cur_list[3] += 1
                elif model_2_confidence < model_1_confidence:
                    cur_list[4] += 1
                else:
                    cur_list[5] += 1
        else:
            pass
    return ans

def data_selection_metrics(metric_name, p1_list, p2_list):
    if metric_name == "Margin":
        # p1_top2_list = heapq.nlargest(2, range(len(p1_list)), p1_list.__getitem__)
        # p2_top2_list = heapq.nlargest(2, range(len(p2_list)), p2_list.__getitem__)  
        p1_top2 = heapq.nlargest(2,p1_list)
        p2_top2 = heapq.nlargest(2,p2_list)
        
        metric_1 = p1_top2[0]-p1_top2[1]
        metric_2 = p2_top2[0]-p2_top2[1]
        return metric_1, metric_2
    else:
        raise Exception("没有选择度量")


def generate_generator_multiple(generator_left, generator_right, data_frame, batch_size, classes, target_size_1, target_size_2):
    '''
    target_size = (224,224)
    '''
    # simulation
    genX1=generator_left.flow_from_dataframe(data_frame, x_col='file_path', y_col='label', target_size=target_size_1, class_mode='categorical',
                                    color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size)  
                                                                                                                # weather:rgb  150, 150

    genX2=generator_right.flow_from_dataframe(data_frame, x_col='file_path', y_col='label', target_size=target_size_2, class_mode='categorical',
                                    color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size)  # weather:rgb 200, 400

    # sports
    # genX1=generator_left.flow_from_dataframe(data_frame, x_col='filepaths', y_col='labels', target_size=(224,224), class_mode='categorical',
    #                                 color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size)   
    #                                                                                                             # weather:rgb  150, 150

    # genX2=generator_right.flow_from_dataframe(data_frame, x_col='filepaths', y_col='labels', target_size=(224,224), class_mode='categorical',
    #                                 color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size)  # weather:rgb 200, 400

    # weather
    # genX1=generator_left.flow_from_dataframe(data_frame, x_col='file_path', y_col='label', target_size=target_size, class_mode='categorical',
    #                                 color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size)   
                                                                                                                

    # genX2=generator_right.flow_from_dataframe(data_frame, x_col='file_path', y_col='label', target_size=target_size, class_mode='categorical',
    #                                 color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size)  
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label

def exceptionInstanceSplit(csv_path, queue_excepiton_path):
    '''
    从异常队列中得到每个实例的model_A的对其度量, 和 model_B的对其度量。以及该样本的真实source
    '''
    def help(str_data):
        ans_list = []
        data_list = re.split("\s+",str_data)
        data_list[0] = data_list[0].split("[")[1]
        data_list[-1] = data_list[-1].split("]")[0]
        for data in data_list:
            if data == '':
                continue
            ans_list.append(np.float32(data))
        return ans_list
    merged_df = pd.read_csv(csv_path) 
    queue_exception = joblib.load(queue_excepiton_path)
    x_axis = []
    y_axis = []
    source_list = []
    all_error = []
    label_list = []
    while not queue_exception.empty():
        priority, index = queue_exception.get()
        row = merged_df.iloc[index]
        probabilty_partyOne_list = row["probabilty_partyOne_list"]
        probabilty_partyOne_list = help(probabilty_partyOne_list)
        probabilty_partyTwo_list = row["probabilty_partyTwo_list"]
        probabilty_partyTwo_list = help(probabilty_partyTwo_list)
        source = row["source"]
        label = row["label"]
        model_1_predic_isTrue = row["model_1_predic_isTrue"]
        model_2_predic_isTrue = row["model_2_predic_isTrue"]
        label_list.append(label)
        if model_1_predic_isTrue == 0 and model_2_predic_isTrue == 0:
            all_error.append(1)
        else:
            all_error.append(0)
        metric_1, metric_2 = data_selection_metrics("Margin", probabilty_partyOne_list, probabilty_partyTwo_list)
        x_axis.append(metric_1)
        y_axis.append(metric_2)
        source_list.append(source)
    return x_axis, y_axis, source_list, all_error, label_list
'''         
上面组织好了
下面都是 补充 应用了
'''
def fromwhere(csv_path, extend_row_name, save_dir, save_fileName):
    '''
    extend_row_name: "source"
    save_dir: "./saved/custom_split/shape/Geometric_Shapes_Mathematics/merge"

    '''
    merged_df = pd.read_csv(csv_path)
    list_source = []
    for i in range(750):
        list_source.append(1)
    for i in range(750, 1750):
        list_source.append(2)
    
    source_Series = pd.Series(list_source, name=extend_row_name)
    merged_df = pd.concat([merged_df, source_Series], axis=1)
    print(merged_df)
    save_path = os.path.join(save_dir, save_fileName)
    merged_df.to_csv(save_path) #  ignore_index=True
    print("fromwhere() success")

def mapping(dir_dataset_1, dir_dataset_1_overlap, dir_dataset_2, dir_dataset_2_overlap):
    '''
    获得两个数据集overlap label 的映射
    args:
        dir_dataset_1: "../dataSets/overlap_datasets/weather/Weather_dataset/dataset_pure_split/val"
        dir_dataset_1_overlap: "../dataSets/overlap_datasets/weather/overlap_dataset/Weather_dataset"
        dir_dataset_2: "../dataSets/overlap_datasets/weather/Weather_Image_Recognition/dataset_pure_split/val"
        dir_dataset_2_overlap: "../dataSets/overlap_datasets/weather/overlap_dataset/Weather_Image_Recognition"
    '''
    dir_className_1 = os.listdir(dir_dataset_1)
    dir_className_1 = deleteIgnoreFile(dir_className_1)
    dir_className_1.sort()

    
    dir_dataset_1_overlap = os.listdir(dir_dataset_1_overlap)
    dir_dataset_1_overlap = deleteIgnoreFile(dir_dataset_1_overlap)
    dir_dataset_1_overlap.sort()

    index_list_1 = []
    for className in dir_dataset_1_overlap:
        index_list_1.append(dir_className_1.index(className))


    
    dir_className_2 = os.listdir(dir_dataset_2)
    dir_className_2 = deleteIgnoreFile(dir_className_2)
    dir_className_2.sort()

    dir_dataset_2_overlap = os.listdir(dir_dataset_2_overlap)
    dir_dataset_2_overlap = deleteIgnoreFile(dir_dataset_2_overlap)
    dir_dataset_2_overlap.sort()

    index_list_2 = []
    for className in dir_dataset_2_overlap:
        index_list_2.append(dir_className_2.index(className))

    assert len(index_list_1) == len(index_list_2), "overlap标签没有对应上"

    dic_oneToTwo = {}
    for label_1, label_2 in zip(index_list_1, index_list_2):
        dic_oneToTwo[label_1] = label_2
    
    return dic_oneToTwo

def isOverlap_groundTrue(csv_path, data_path, save_dir, save_fileName):
    '''
    标记这个实例groundTrue是否是overlap
    data_path: "saved/"+scene+"/val/overlap_label_map/dic_oneToTwo.data"
    '''
    merged_df = pd.read_csv(csv_path)
    dic_oneToTwo = joblib.load(data_path)
    isOverlap_groundTrue_list = []
    for index, row in merged_df.iterrows():
        if row["source"] == 1 and row["local_label_index"] in dic_oneToTwo.keys():
            isOverlap_groundTrue_list.append(1)
        elif row["source"] == 2 and row["local_label_index"] in dic_oneToTwo.values():
            isOverlap_groundTrue_list.append(1)
        else:
            isOverlap_groundTrue_list.append(0)
    isOverlap_groundTrue = pd.Series(isOverlap_groundTrue_list, name='isOverlap_groundTrue')
    merged_df = pd.concat([merged_df, isOverlap_groundTrue], axis=1)
    # print(merged_df)
    save_path = os.path.join(save_dir, save_fileName)
    merged_df.to_csv(save_path, index=False)
    print("isOverlap_groundTrue() success")

def get_overlap_merged_df(csv_path, save_dir, save_fileName):
    merged_df = pd.read_csv(csv_path)
    overlap_merged_df = merged_df[merged_df["isOverlap_groundTrue"] == 1]
    save_path = os.path.join(save_dir, save_fileName)
    overlap_merged_df.to_csv(save_path, index=False)
    print("get_overlap_merged_df() success")

def get_unique(csv_path, source, save_dir, save_fileName):
    '''
    source: 1>A | 2>B
    save_fileName: A_unique_merged_df.csv
    '''
    merged_df = pd.read_csv(csv_path)
    # A_unique_merged_df = merged_df[merged_df["isOverlap_groundTrue"] == 0 & merged_df["source"] == 0]
    unique_merged_df = merged_df.loc[(merged_df['isOverlap_groundTrue'] == 0) & (merged_df['source'] == source)]
    # print(A_unique_merged_df)
    # B_unique_merged_df = merged_df[merged_df["isOverlap_groundTrue"] == 0 & merged_df["source"] == 1]
    # B_unique_merged_df = merged_df.loc[(merged_df['isOverlap_groundTrue'] == 0) & (merged_df['source'] == 2)]
    save_path = os.path.join(save_dir, save_fileName)
    unique_merged_df.to_csv(save_path, index=False)
    # B_unique_merged_df.to_csv("./saved/"+scene+"/val/B_unique_merged_df.csv", index = False)
    print("get_unique() success")

def sampling(scene, priority_data_version):
    '''
    scene: shapes|sports|weather
    priority_data_version:"priority_data_version_1"
    '''
    if priority_data_version == "priority_data_version_1":
        queue_consistence_overlap = joblib.load("./saved/"+scene+"/val/data"+priority_data_version+"/queue_consistence_overlap.data")
        queue_consistence_unOverlap = joblib.load("./saved/"+scene+"/val/data"+priority_data_version+"/queue_consistence_unOverlap.data")
        queue_unConsistence = joblib.load("./saved/"+scene+"/val/data"+priority_data_version+"/queue_unConsistence.data")
        print(queue_consistence_overlap.qsize())
        print(queue_consistence_unOverlap.qsize())
        print(queue_unConsistence.qsize())
        print(queue_consistence_overlap.qsize() + queue_consistence_unOverlap.qsize() + queue_unConsistence.qsize())
    elif priority_data_version == "priority_data_version_2":
        queue_consistence_overlap = joblib.load("./saved/"+scene+"/val/data/"+priority_data_version+"/queue_consistence_overlap.data")
        queue_consistence_unOverlap = joblib.load("./saved/"+scene+"/val/data/"+priority_data_version+"/queue_consistence_unOverlap.data")
        queue_unConsistence_allOverlap = joblib.load("./saved/"+scene+"/val/data/"+priority_data_version+"/queue_unConsistence_allOverlap.data")
        queue_unConsistence_allUnique = joblib.load("./saved/"+scene+"/val/data/"+priority_data_version+"/queue_unConsistence_allUnique.data")
        queue_unConsistence_existOverlap = joblib.load("./saved/"+scene+"/val/data/"+priority_data_version+"/queue_unConsistence_existOverlap.data")
        a = queue_consistence_overlap.qsize()
        b = queue_consistence_unOverlap.qsize()
        c = queue_unConsistence_allOverlap.qsize()
        d = queue_unConsistence_allUnique.qsize()
        e = queue_unConsistence_existOverlap.qsize()
        print("queue_consistence_overlap:",a)
        print("queue_consistence_unOverlap",b)
        print("queue_unConsistence_allOverlap",c)
        print("queue_unConsistence_allUnique",d)
        print("queue_unConsistence_existOverlap",e)
        print("total",a+b+c+d+e)





if __name__ == '__main__':
    '''
        filepath, label, local_label_index, 
        source,
        model_1_predict_label,
        model_2_predict_label,
        model_1_confidence,
        model_2_confidence,
        model_1_predict_label_isOverlap,
        model_2_predict_label_isOverlap,
        isConsistence,
        isOverlap_groundTrue,
        global_label_index,
        model_1_predic_isTrue,
        model_2_predic_isTrue

    '''

    '''得到各自的数据集dataFrame'''

    # dir_path = "/data/mml/dataSets/overlap_datasets/custom_dataset/train_test/party_B/percent_20_adv/dataset/test"
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/party_B/test"
    # save_file_name = "test.csv"

    # animal 
    # source = 1
    # dic_localLabel_to_globalLable = {0:1, 1:3, 2:4, 3:5, 4:6}
    # source = 2
    # dic_localLabel_to_globalLable = {0:0, 1:2, 2:4, 3:7}  # 必要的话需要写个函数

    # # weather
    # source = 1
    # dic_localLabel_to_globalLable = {0:0, 1:1, 2:5, 3:7, 4:9}
    # source = 2
    # dic_localLabel_to_globalLable = {0:2, 1:3, 2:4, 3:5, 4:6, 5:8}  # 必要的话需要写个函数
    
    # car_body_style
    # source = 1
    # dic_localLabel_to_globalLable = {0:0, 1:1, 2:2, 3:4}

    # source = 2
    # dic_localLabel_to_globalLable = {0:3, 1:4, 2:5, 3:6}
    
    # custom_dataset_shape
    # source = 1
    # dic_localLabel_to_globalLable = {0:0, 1:1, 2:2}

    # source = 2
    # dic_localLabel_to_globalLable = {0:2, 1:3, 2:4, 3:5}

    # gen_df(dir_path, save_dir, save_file_name, source, dic_localLabel_to_globalLable)

    
    '''将各自的数据集dataFrame合并'''
    # csv_1_filePath = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/party_A/test/test.csv"
    # csv_2_filePath = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/party_B/test/test.csv"
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test"
    # save_fileName = "merged_df.csv"
    # concat(csv_1_filePath, csv_2_filePath, save_dir, save_fileName)

    '''overlap label mapping'''
    # sameName()

    '''统计各个样本的预测结果 伪标签 和 置信度'''
    # merged_csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test/merged_df.csv"  # "./saved/real/animal/merge/val/merged_df.csv"
    # custom_dataset shape
    # model_1 = load_model("/data/mml/dataSets/overlap_datasets/custom_dataset/train_test/party_A/percent_20_adv/models/model_007_0.9960.h5")
    # model_2 = load_model("/data/mml/dataSets/overlap_datasets/custom_dataset/train_test/party_B/percent_20_adv/models/model_035_0.9940.h5")
    # model_1 = load_model("/home/mml/workspace/dataSets/overlap_datasets/weather/Weather_dataset/saved/model/model_014_0.9766.h5")
    # model_2 = load_model("../dataSets/overlap_datasets/weather/Weather_Image_Recognition/saved/model/cut_rainy/model_010_0.9589.h5")
    # model_1 = load_model("/data/mml/dataSets/overlap_datasets/animal/train_test/party_A/models/model_004_0.9439.h5")
    # model_2 = load_model("/data/mml/dataSets/overlap_datasets/animal/train_test/party_B/models/model_012_0.9721.h5")
    # model_1 = load_model("/data/mml/dataSets/overlap_datasets/car_body_style/train_test/party_A/models/model_025_0.9507.h5")
    # model_2 = load_model("/data/mml/dataSets/overlap_datasets/car_body_style/train_test/party_B/models/model_024_0.8900.h5")

    # col_1_name = "model_1_predict_label"
    # col_2_name = "model_1_confidence"
    # col_3_name = "model_2_predict_label"
    # col_4_name = "model_2_confidence"

    # 这个要变
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test"
    # save_fileName = "merged_predic_df.csv"

    # car_body_style: 
        # target_size_1 = (224,224) target_size_2 = (224,224)
    # animal
        # target_size_1 = (224,224) target_size_2 = (224,224)
    #custom_split_shape_Geometric_Shapes_Mathematics
        # target_size_1 = (224,224) target_size_2 = (224, 224)

    # target_size_1 = (224, 224) 
    # target_size_2 =  (224, 224)
    # predic_ans(merged_csv_path, model_1, model_2, col_1_name, col_2_name, col_3_name, col_4_name,
    #     save_dir, save_fileName, target_size_1, target_size_2)

    
    # image = tf.keras.preprocessing.image.load_img(
    #     "/data/mml/dataSets/overlap_datasets/animal/Animal_5_Mammal/dataSet_cut/dataset/tvt/val/elephant/P6002Y9T9CF0.jpg", grayscale=False, color_mode='rgb', target_size=(224, 224),
    #     # interpolation='nearest'
    #     )


    # image_array = np.array(image, dtype="float32")        
    # input = np.expand_dims(image_array, 0) 

    # output_1 = model_1(input)
    # output_2 = model_1.predict(input)
    # output_3 = model_1.predict_generator(input)
    # print("model_1:output_1",output_1.numpy())
    # print("model_1:output_2",output_2)
    # print("model_1:output_3",output_3)

    # output_1 = model_2(input)
    # output_2 = model_2.predict(input)
    # output_3 = model_2.predict_generator(input)
    # print("model_2:output_1",output_1.numpy())
    # print("model_2:output_2",output_2)
    # print("model_2:output_3",output_3)
    # print("jfaljla")
    '''
    统计预测是否对
    '''
    # merged_csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test/merged_predic_df.csv"

    # Geometric_Shapes_Mathematics
    # model_1_localLabel_to_globalLable = {0:0, 1:1, 2:2}
    # model_2_localLabel_to_globalLable = {0:2, 1:3, 2:4, 3:5}

    # # weather
    # # model_1_localLabel_to_globalLable = {0:0, 1:1, 2:5, 3:7, 4:9}
    # # model_2_localLabel_to_globalLable = {0:2, 1:3, 2:4, 3:5, 4:6, 5:8}

    # animal
    # model_1_localLabel_to_globalLable = {0:1, 1:3, 2:4, 3:5, 4:6}
    # model_2_localLabel_to_globalLable = {0:0, 1:2, 2:4, 3:7} 

    # car_body_style
    # model_1_localLabel_to_globalLable = {0:0, 1:1, 2:2, 3:4}
    # model_2_localLabel_to_globalLable = {0:3, 1:4, 2:5, 3:6}

    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test"
    # save_fileName = "merged_predic_trueOrFalse_df.csv"
    # predic_trueOrFalse(merged_csv_path, model_1_localLabel_to_globalLable, model_2_localLabel_to_globalLable, save_dir, save_fileName)

    '''统计伪标签是否是overlap'''
    # dir_path = "/data/mml/dataSets/overlap_datasets/custom_dataset/train_test/party_B/percent_20_adv/dataset/test"
    # dir_path_overlap = "/data/mml/dataSets/overlap_datasets/custom_dataset/train_test/overlap/percent_20/party_B/test"
    # csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test/merged_predic_trueOrFalse_isOverlap_df.csv"
    # col_name = "model_2_predict_label"
    # extend_col_name = "model_2_predict_label_isOverlap"
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test"
    # save_fileName = "merged_predic_trueOrFalse_isOverlap_again_df.csv"
    # is_overlap(dir_path, dir_path_overlap, csv_path, col_name, extend_col_name, save_dir, save_fileName)
    
    '''统计伪标签是否是 一致'''
    # csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test/merged_predic_trueOrFalse_isOverlap_again_df.csv"
    # dir_path_1 = "/data/mml/dataSets/overlap_datasets/custom_dataset/train_test/party_A/percent_20_adv/dataset/test"
    # dir_path_overlap_1 = "/data/mml/dataSets/overlap_datasets/custom_dataset/train_test/overlap/percent_20/party_A/test"
    # dir_path_2 = "/data/mml/dataSets/overlap_datasets/custom_dataset/train_test/party_B/percent_20_adv/dataset/test"
    # dir_path_overlap_2 = "/data/mml/dataSets/overlap_datasets/custom_dataset/train_test/overlap/percent_20/party_B/test"
    # model_1_predict_label = "model_1_predict_label"
    # model_2_predict_label = "model_2_predict_label"
    # extend_col_name = "isConsistence"
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test"
    # save_fileName = "merged_predic_trueOrFalse_isOverlap_again_isConsis_df.csv"
    # isConsistence(csv_path, 
    #     dir_path_1, dir_path_overlap_1,
    #     dir_path_2, dir_path_overlap_2,
    #     model_1_predict_label, model_2_predict_label,
    #     extend_col_name,save_dir, save_fileName)
    
    '''
    dic_oneToTwo
    '''
    # dir_dataset_1 = "/data/mml/dataSets/overlap_datasets/custom_dataset/train_test/party_A/percent_20_adv/dataset/test"
    # dir_dataset_1_overlap = "/data/mml/dataSets/overlap_datasets/custom_dataset/train_test/overlap/percent_20/party_A/test"
    # dir_dataset_2 = "/data/mml/dataSets/overlap_datasets/custom_dataset/train_test/party_B/percent_20_adv/dataset/test"
    # dir_dataset_2_overlap = "/data/mml/dataSets/overlap_datasets/custom_dataset/train_test/overlap/percent_20/party_B/test"
    # dic_oneToTwo = mapping(dir_dataset_1, dir_dataset_1_overlap, dir_dataset_2, dir_dataset_2_overlap)
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/data"
    # save_fileName = "dic_oneToTwo.data"
    # saveData(dic_oneToTwo, save_dir, save_fileName)
    # print("mapping() success")

    '''
    isOverlap_groundTrue
    '''
    # csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test/merged_predic_trueOrFalse_isOverlap_again_isConsis_df.csv"
    # data_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/data/dic_oneToTwo.data"
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test"
    # save_fileName = "merged_predic_trueOrFalse_isOverlap_again_isConsis_isOverlapGT_df.csv"
    # isOverlap_groundTrue(csv_path, data_path, save_dir, save_fileName)

    '''
    shuffle csv
    '''
    # csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test/merged_predic_trueOrFalse_isOverlap_again_isConsis_isOverlapGT_df.csv"
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test"
    # file_name = "merged_predic_trueOrFalse_isOverlap_again_isConsis_isOverlapGT_shuffle_df.csv"
    # shuffle_dataFrame(csv_path, save_dir, file_name)

    '''
    补充combination_model_prob combination_model_confidence, combination_model_probabilty_list
    '''
    # merged_csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test/merged_predic_trueOrFalse_isOverlap_again_isConsis_isOverlapGT_shuffle_df.csv"
    # combination_model = load_model("./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge_model/combination_model_lock_newWeights.h5")
    # col_1_name = "combination_model_predict_label"
    # col_2_name = "combination_model_confidence"
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test"
    # save_fileName = "merged_predic_trueOrFalse_isOverlap_again_isConsis_isOverlapGT_shuffle_combinationInfo_df.csv"
    # target_size_1 = (224, 224)
    # target_size_2 =  (224, 224)
    # combination_model_predict(merged_csv_path, combination_model, col_1_name, col_2_name,
    #         save_dir, save_fileName, target_size_1, target_size_2)
    # print("combination_model_predict() success")

    '''
    A_unique_merged_df.csv
    '''
    # csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test/merged_predic_trueOrFalse_isOverlap_again_isConsis_isOverlapGT_shuffle_combinationInfo_df.csv"
    # source = 2
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test"
    # save_fileName = "B_unique_merged_df.csv"
    # get_unique(csv_path, source, save_dir, save_fileName)

    '''
    overlap_merged_df.csv
    '''
    # csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test/merged_predic_trueOrFalse_isOverlap_again_isConsis_isOverlapGT_shuffle_combinationInfo_df.csv"
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test"
    # save_fileName = "overlap_merged_df.csv"
    # get_overlap_merged_df(csv_path, save_dir, save_fileName)

    '''队列方案_1根据伪标签确定优先级,存入队列'''
    # # queue_consistence_overlap, queue_consistence_unOverlap, queue_unConsistence_allOverlap, queue_unConsistence_existOverlap, queue_unConsistence_allUnique = getQueue(scene)
    # csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/merge/merged_predic_isOverlap_again_isConsistence_df.csv"
    # model_1_confidence = "model_1_confidence"
    # model_2_confidence = "model_2_confidence"
    # model_1_predict_label_isOverlap = "model_1_predict_label_isOverlap"
    # model_2_predict_label_isOverlap = "model_2_predict_label_isOverlap"
    # isConsis = "isConsistence"
    # queue_consistence_overlap, queue_consistence_unOverlap, queue_unConsistence_allOverlap, queue_unConsistence_existOverlap, queue_unConsistence_allUnique = getQueue(csv_path,
    #                 model_1_confidence, model_2_confidence,
    #                 model_1_predict_label_isOverlap,model_2_predict_label_isOverlap,
    #                 isConsis)
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/data"
    # saveData(queue_consistence_overlap, save_dir+"/queue_consistence_overlap.data")
    # saveData(queue_consistence_unOverlap, save_dir+"/queue_consistence_unOverlap.data")
    # saveData(queue_unConsistence_allOverlap, save_dir+"/queue_unConsistence_allOverlap.data")
    # saveData(queue_unConsistence_existOverlap, save_dir+"/queue_unConsistence_existOverlap.data")
    # saveData(queue_unConsistence_allUnique, save_dir+"/queue_unConsistence_allUnique.data")
    # queue_consistence_overlap = joblib.load(save_dir+"/queue_consistence_overlap.data")
    # queue_consistence_unOverlap = joblib.load(save_dir+"/queue_consistence_unOverlap.data")
    # queue_unConsistence_allOverlap = joblib.load(save_dir+"/queue_unConsistence_allOverlap.data")
    # queue_unConsistence_allUnique = joblib.load(save_dir+"/queue_unConsistence_allUnique.data")
    # queue_unConsistence_existOverlap = joblib.load(save_dir+"/queue_unConsistence_existOverlap.data")
    # a = queue_consistence_overlap.qsize()
    # b = queue_consistence_unOverlap.qsize()
    # c = queue_unConsistence_allOverlap.qsize()
    # d = queue_unConsistence_allUnique.qsize()
    # e = queue_unConsistence_existOverlap.qsize()
    # print("queue_consistence_overlap:",a)
    # print("queue_consistence_unOverlap",b)
    # print("queue_unConsistence_allOverlap",c)
    # print("queue_unConsistence_allUnique",d)
    # print("queue_unConsistence_existOverlap",e)
    # print("total",a+b+c+d+e)
    # print("main success")
    
    '''
    队列方案_2根据伪标签确定优先级,存入队列
    '''
    # csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/merge/val/merged_predic_isOverlap_again_isConsistence_fromWhere_isOverlap_groundTrue_df.csv"
    # model_1_confidence = "model_1_confidence"
    # model_2_confidence = "model_2_confidence"
    # isConsis = "isConsistence"
    # queue_not_consistence, queue_consistence = getQueue_2(csv_path,
    #             model_1_confidence, model_2_confidence,
    #             isConsis)
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/data/queue_scheme_2"
    # saveData(queue_not_consistence, save_dir, "queue_not_consistence.data")
    # saveData(queue_consistence, save_dir, "queue_consistence.data")
    # queue_not_consistence = joblib.load(save_dir+"/queue_not_consistence.data")
    # queue_consistence = joblib.load(save_dir+"/queue_consistence.data")
    # a = queue_not_consistence.qsize()
    # b = queue_consistence.qsize()
    # print("queue_not_consistence:",a)
    # print("queue_consistence",b)
    # print("total",a+b)

    '''
    队列方案_3 
    '''
    # csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/merge/val/merged_predic_isOverlap_again_isConsistence_fromWhere_isOverlap_groundTrue_df.csv"
    # queue_consistence, queue_adversary, queue_exception ,queue_adversary_exception = getQueue_3(csv_path)
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/data/queue_scheme_3"
    # saveData(queue_consistence, save_dir, "queue_consistence.data")
    # saveData(queue_adversary, save_dir, "queue_adversary.data")
    # saveData(queue_exception, save_dir, "queue_exception.data")
    # saveData(queue_adversary_exception, save_dir, "queue_adversary_exception.data")

    # queue_consistence = joblib.load(save_dir+"/queue_consistence.data")
    # queue_adversary = joblib.load(save_dir+"/queue_adversary.data")
    # queue_exception = joblib.load(save_dir+"/queue_exception.data")
    # queue_adversary_exception = joblib.load(save_dir+"/queue_adversary_exception.data")
    # a = queue_consistence.qsize()
    # b = queue_adversary.qsize()
    # c = queue_exception.qsize()
    # d = queue_adversary_exception.qsize()

    # print("queue_consistence:",a)
    # print("queue_adversary",b)
    # print("queue_exception", c)
    # print("queue_adversary_exception", d)
    # print("total",a+b+c+d)

    '''
    队列方案_4
    '''
    # csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/merge/val/merged_predic_isOverlap_again_isConsistence_fromWhere_isOverlap_groundTrue_df.csv"
    # merged_df = pd.read_csv(csv_path)  
    # classes_array = merged_df["labels"].unique()
    # label_space = classes_array.tolist()
    # label_AToGlobal = {0:0, 1:1, 2:2}
    # label_BToGlobal = {0:2, 1:3, 2:4, 3:5}
    # queue_list, queue_ambiguous = getQueue_4(csv_path, label_space, label_AToGlobal, label_BToGlobal)
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/data/queue_scheme_4"
    # for i in range(len(queue_list)):
    #     queue = queue_list[i]
    #     saveData(queue, save_dir, "queue_"+str(i)+".data")
    # saveData(queue_ambiguous, save_dir, "queue_ambiguous.data")

    # queue_list = []
    # for i in range(len(label_space)):
    #     queue_list.append(joblib.load(save_dir+"/queue_"+str(i)+".data"))
    # queue_ambiguous = joblib.load(save_dir+"/queue_ambiguous.data")
    # total = 0
    # for i in range(len(queue_list)):
    #     cur_size = queue_list[i].qsize()
    #     total += cur_size
    #     print(i, queue_list[i].qsize())
    # total += queue_ambiguous.qsize()
    # print("ambiguous", queue_ambiguous.qsize())
    # print("total", total)

    '''
    队列方案_5
    '''
    # csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test/merged_predic_trueOrFalse_isOverlap_again_isConsis_isOverlapGT_shuffle_combinationInfo_df.csv"
    # queue_agree, queue_adversary, queue_adv_A_ex_B, queue_adv_B_ex_A, queue_exception = get_Queue5(csv_path)
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/data/queue_scheme_5"

    # saveData(queue_agree, save_dir, "queue_agree.data")
    # saveData(queue_adversary, save_dir, "queue_adversary.data")
    # saveData(queue_adv_A_ex_B, save_dir, "queue_adv_A_ex_B.data")
    # saveData(queue_adv_B_ex_A, save_dir, "queue_adv_B_ex_A.data")
    # saveData(queue_exception, save_dir, "queue_exception.data")


    # queue_agree = joblib.load(save_dir+"/queue_agree.data")
    # queue_adversary = joblib.load(save_dir+"/queue_adversary.data")
    # queue_adv_A_ex_B = joblib.load(save_dir+"/queue_adv_A_ex_B.data")
    # queue_adv_B_ex_A = joblib.load(save_dir+"/queue_adv_B_ex_A.data")
    # queue_exception = joblib.load(save_dir+"/queue_exception.data")
    # total = 0
    # a = queue_agree.qsize()
    # b = queue_adversary.qsize()
    # c = queue_adv_A_ex_B.qsize()
    # d = queue_adv_B_ex_A.qsize()
    # e = queue_exception.qsize()
    # total = a+b+c+d+e
    # print("queue_agree", a)
    # print("queue_adversary", b)
    # print("queue_adv_A_ex_B", c)
    # print("queue_adv_B_ex_A", d)
    # print("queue_exception", e)
    # print("total", total)

    '''
    保存分组的mcp_priority
    '''
    # merged_df = pd.read_csv("saved/real/animal/train_test/merge/test/merged_predic_trueOrFalse_isOverlap_again_isConsis_isOverlapGT_shuffle_combinationInfo_df.csv")
    
    # queue_agree = joblib.load("saved/real/animal/train_test/data/queue_scheme_5/queue_agree.data")
    # queue_adv_A_ex_B = joblib.load("saved/real/animal/train_test/data/queue_scheme_5/queue_adv_A_ex_B.data")
    # queue_adv_B_ex_A =  joblib.load("saved/real/animal/train_test/data/queue_scheme_5/queue_adv_B_ex_A.data")
    # queue_exception = joblib.load("saved/real/animal/train_test/data/queue_scheme_5/queue_exception.data")

    # queue_agree_mcp, queue_adv_A_exp_B_mcp, queue_adv_B_exp_A_mcp, queue_exp_mcp = group_mcp(merged_df, queue_agree, queue_adv_A_ex_B, queue_adv_B_ex_A, queue_exception)

    # save_dir = "saved/real/animal/train_test/data/queue_scheme_5/mcp_priority"
    # saveData(queue_agree_mcp, save_dir, "queue_agree_mcp.data")
    # saveData(queue_adv_A_exp_B_mcp, save_dir, "queue_adv_A_exp_B_mcp.data")
    # saveData(queue_adv_B_exp_A_mcp, save_dir, "queue_adv_B_exp_A_mcp.data")
    # saveData(queue_exp_mcp, save_dir, "queue_exp_mcp.data")
    # print("group_mcp() success")

    '''
    DeepGini队列获取
    '''
    # csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test/merged_predic_trueOrFalse_isOverlap_again_isConsis_isOverlapGT_shuffle_combinationInfo_df.csv"
    # queue = get_DeepGiniQueue(csv_path, measure=None, algorithm_flag="combination")
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/data/DeepGiniQueue/combination"
    # saveData(queue, save_dir, "queue.data")
    # queue = joblib.load(os.path.join(save_dir, "queue.data"))
    # print("total:{}".format(queue.qsize()))

    '''
    MCP 队列获取
    '''
    # csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge/test/merged_predic_trueOrFalse_isOverlap_again_isConsis_isOverlapGT_shuffle_combinationInfo_df.csv"
    # queue_matrix = get_MCPQueue(csv_path, source = None, algorithm_flag="combination")
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/data/MCP_Queue/combination"
    # # saveData(queue_matrix, save_dir, "queue_matrix_party_{}.data".format(source))
    # saveData(queue_matrix, save_dir, "queue_matrix.data")
    # # 加载回来, 简单统计验证一下
    # queue_matrix = joblib.load(os.path.join(save_dir, "queue_matrix.data"))
    # count = 0
    # for row_index in range(len(queue_matrix)):
    #     for col_index in range(len(queue_matrix[0])):
    #         cur_queue = queue_matrix[row_index][col_index]
    #         count += cur_queue.qsize()
    # print("total:{}".format(count))

    '''
    看看unique c_A和c_B的一个比较情况
    '''
    # csv_path = "./saved/real/animal/merge/val/merged_predic_trueOrFalse_isOverlap_again_isConsis_isOverlapGT_shuffle_df.csv"
    # unique_label_list_A = ["cat", "dog", "horse", "lion"]
    # unique_label_list_B = ["butterfly", "chicken", "spider"]
    # ans = statistic_unique_Label(csv_path, unique_label_list_A, unique_label_list_B)
    # total = 0
    # for key in ans.keys():
    #     cur_list = ans.get(key)
    #     total += sum(cur_list)
    #     print(key, cur_list)
    # print(total)

    ### sampling(scene, priority_data_version)

    ''''
    下边都是应用了
    '''
    # csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/merge/merged_predic_isOverlap_again_isConsistence_df.csv"
    # extend_row_name = "source"
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/merge"
    # save_fileName = "merged_predic_isOverlap_again_isConsistence_fromWhere_df.csv"
    # fromwhere(csv_path, extend_row_name, save_dir, save_fileName)

    '''
    添加gloabl_label 列
    '''
    # csv_path = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/merge/val/merged_predic_isOverlap_again_isConsistence_fromWhere_isOverlap_groundTrue_df.csv"
    # part_A_labelToGlobal = {0:0, 1:1, 2:2}
    # part_B_labelToGlobal = {0:2, 1:3, 2:4, 3:5}
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/merge/val"
    # save_fileName = "merged_predic_isOverlap_again_isConsistence_fromWhere_isOverlap_groundTrue_globalLabel_df.csv"
    # global_label_index(csv_path, part_A_labelToGlobal, part_B_labelToGlobal, save_dir, save_fileName)

    '''
    看能不能区分source
    '''
    # csv_path = "./saved/simulation/percent_5/merge/val/merged_predic_trueOrFalse_isOverlap_again_isConsis_isOverlapGT_shuffle_df.csv"
    # queue_excepiton_path = "./saved/simulation/percent_5/data/queue_exception.data"
    # x_axis, y_axis, source_list, all_error, label_list = exceptionInstanceSplit(csv_path,queue_excepiton_path)                                                                
    # point = pd.DataFrame({'x_axis':x_axis,'y_axis':y_axis, "source":source_list, "label":label_list})

    # source_1_df = point[(point['source'] == 1)]
    # print(source_1_df['label'].value_counts())

    # source_1_df = point[(point['source'] == 2)]
    # print(source_1_df['label'].value_counts())

    # point.to_csv("./saved/simulation/percent_5/data/point.csv")
    # x_1 = []
    # y_1 = []
    # x_2 = []
    # y_2 = []
    # x_error = []
    # y_error = []
    # for i in range(len(source_list)):
    #     source = source_list[i]
    #     isError = all_error[i]
    #     if isError == 1:
    #         x_error.append(x_axis[i])
    #         y_error.append(y_axis[i])
    #     else:
    #         if source == 1:
    #             x_1.append(x_axis[i])
    #             y_1.append(y_axis[i])
    #         else:
    #             x_2.append(x_axis[i])
    #             y_2.append(y_axis[i])
    # plt.xlabel('model_A metric')
    # plt.ylabel('model_B metric')
    # plt.title("exception queue data")
    # s1 = plt.scatter(x_1, y_1, color = 'red', label='A')
    # s2 = plt.scatter(x_2, y_2, color = 'green', label='B')
    # s3 = plt.scatter(x_error, y_error, color = 'blue', label='Error')
    # plt.text(0.2, 0.4, "Error ratio:"+str(round(sum(all_error)/len(all_error),4)), size = 15, alpha = 0.5, color='r')
    # plt.legend()
    # plt.show(block=False)
    pass    

