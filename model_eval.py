from tensorflow.keras.models import Model, Sequential, load_model
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.losses import CategoricalCrossentropy
import joblib
sys.path.append("/home/mml/workspace/model_reuse_v2/")
from utils import deleteIgnoreFile
from DataSetConfig import car_body_style_config, flower_2_config, food_config, fruit_config, sport_config, weather_config, animal_config, animal_2_config, animal_3_config
import Base_acc

def generate_generator_multiple(batches_A, batches_B):
    '''
    将连个模型的输入bath 同时返回
    '''
    # simulation

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
        X1i = batches_A.next()
        X2i = batches_B.next()
        yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label

def getClasses(dir_path):
    '''
    得到分类目录的分类列表
    '''
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    classes_name_list.sort()  # 字典序
    return classes_name_list

def eval_singleModel(config):
    '''
    评估各方原始模型
    '''
    model = load_model(config["model_B_struct_path"])
    if not config["model_B_weight_path"] is None:
        model.load_weights(config["model_B_weight_path"])
    df = pd.read_csv(config["df_eval_party_B_path"])
    batch_size = 32
    gen = config["generator_B_test"]
    classes= getClasses(config["dataset_B_train_path"]) # sorted
    target_size = target_size_B
    # 样本batch
    batches = gen.flow_from_dataframe(
                                    df, 
                                    directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                    x_col='file_path', y_col='label', 
                                    target_size=target_size, class_mode='categorical',
                                    color_mode='rgb', classes = classes, 
                                    shuffle=False, batch_size=batch_size,
                                    validate_filenames=False
                                    )
    print("评估集样本数: {}".format(df.shape[0]))
    # 开始评估
    model.evaluate_generator(generator=batches,steps=batches.n / batch_size, verbose = 1)

def eval_singleExtendModel():
    '''
    评估各方原始模型
    '''
    # 加载模型
    model_path = "/data/mml/overlap_v2_datasets/weather/merged_model/model_B_extended.h5"
    model = load_model(model_path)
    # 编译模型（需要看模型准备是怎么编译的）importent
    model.compile(optimizer='adam',loss=CategoricalCrossentropy(),metrics=['accuracy'])
    # 加载评估集
    csv_path = "/data/mml/overlap_v2_datasets/weather/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv"
    df = pd.read_csv(csv_path) 
    test_gen = ImageDataGenerator(rescale=1./255)   # 归一化 importent
    # 样本分类,告诉生成器模型关注哪些分类
    merged_csv_path = "/data/mml/overlap_v2_datasets/weather/merged_data/train/merged_df.csv"
    merged_df = pd.read_csv(merged_csv_path)
    # 全局类别,字典序列
    classes = merged_df["label"].unique()
    classes = np.sort(classes).tolist()

    target_size = (256,256) # importent
    batch_size = 32
    # 样本batch
    test_batches = test_gen.flow_from_dataframe(
                                                df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size, class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size,
                                                validate_filenames=False
                                                )
    print("评估集样本数: {}".format(df.shape[0]))
    # 开始评估
    model.evaluate_generator(generator=test_batches,steps=test_batches.n / batch_size, verbose = 1)

def eval_combination_Model(config,df):
    '''
    对combination_Model进行评估
    '''
    # 加载 合并模型
    combination_model = load_model(config["combination_model_path"])
    # 加载 合并的训练集csv
    merged_train_df = pd.read_csv(config["merged_train_df"])
    # 模型编译
    combination_model.compile(optimizer="adam",loss="categorical_crossentropy",metrics="accuracy")
    # 加载评估集
    eval_df = df

    test_gen_left = config["generator_A_test"]
    test_gen_right = config["generator_B_test"]

    batch_size = 32
    # 全局类别,字典序列
    classes = merged_train_df["label"].unique()
    classes = np.sort(classes).tolist()

    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    prefix_path = "/data/mml/overlap_v2_datasets/"
    test_batches_A = test_gen_left.flow_from_dataframe(eval_df, 
                                                directory = prefix_path, # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size_A, class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size,
                                                validate_filenames=False)
                                                                                                                # weather:rgb  150, 150

    test_batches_B = test_gen_right.flow_from_dataframe(eval_df, 
                                                directory = prefix_path, # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size_B, class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size,
                                                validate_filenames=False)

    test_batches = generate_generator_multiple(test_batches_A, test_batches_B)

    print("评估集样本数: {}".format(eval_df.shape[0]))
    acc = combination_model.evaluate(test_batches, batch_size = batch_size, verbose=1,steps = test_batches_A.n/batch_size, return_dict=True)
    return acc

def eval_agree(df, model):
    '''
    评估各方原始模型
    '''
    # 加载模型
    model_path = "/data/mml/overlap_v2_datasets/food/merged_model/model_A_extended.h5"
    model = load_model(model_path)
    # 编译模型（需要看模型准备是怎么编译的）importent
    model.compile(optimizer='adam',loss=CategoricalCrossentropy(),metrics=['accuracy'])
    # 加载评估集
    csv_path = "/data/mml/overlap_v2_datasets/food/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv"
    df = pd.read_csv(csv_path) 
    test_gen = ImageDataGenerator()   # 归一化 importent
    # 样本分类,告诉生成器模型关注哪些分类
    merged_csv_path = "/data/mml/overlap_v2_datasets/food/merged_data/train/merged_df.csv"
    merged_df = pd.read_csv(merged_csv_path)
    # 全局类别,字典序列
    classes = merged_df["label"].unique()
    classes = np.sort(classes).tolist()

    target_size = (256,256) # importent
    batch_size = 32
    # 样本batch
    test_batches = test_gen.flow_from_dataframe(
                                                df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size, class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size,
                                                validate_filenames=False
                                                )
    print("评估集样本数: {}".format(df.shape[0]))
    # 开始评估
    model.evaluate_generator(generator=test_batches,steps=test_batches.n / batch_size, verbose = 1)

def eval_stuModel(config):
    '''
    对combination_Model进行评估
    '''
    # 加载 合并模型
    combination_model = load_model(config["stu_model_path"])
    # 加载 合并的训练集csv
    merged_df = pd.read_csv(config["merged_train_df"])
    # 模型编译
    combination_model.compile(optimizer="adam",loss="categorical_crossentropy",metrics="accuracy")
    # 加载评估集
    csv_path = config["merged_df_path"]
    df = pd.read_csv(csv_path) 

    test_gen = ImageDataGenerator(rescale=1./255)

    batch_size = 32
    # 全局类别,字典序列
    classes = merged_df["label"].unique()
    classes = np.sort(classes).tolist()

    target_size = (224,224)
    prefix_path = "/data/mml/overlap_v2_datasets/"
    test_batches = test_gen.flow_from_dataframe(df, 
                                            directory = prefix_path, # 添加绝对路径前缀
                                            x_col='file_path', y_col='label', 
                                            target_size=target_size, class_mode='categorical',
                                            color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size,
                                            validate_filenames=False)
                                                                                                                # weather:rgb  150, 150


    print("评估集样本数: {}".format(df.shape[0]))
    ans = combination_model.evaluate(test_batches, batch_size = batch_size, verbose=1,steps = test_batches.n/batch_size, return_dict=True)
    acc = ans["accuracy"]
    loss = ans["loss"]
    return acc, loss

if __name__ == "__main__":
    print("=====model_eval.py=======")
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    config = car_body_style_config
    # eval_combination_Model(config)
    # eval_singleModel()
    # eval_singleExtendModel()
    # eval_stuModel()
    pass