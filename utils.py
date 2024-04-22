'''
1:工具文件:比如deleteIgnoreFile,makedir_help,saveData
2:原生数据集组织与划分
'''

import os
import io
import re
import math
import joblib
import splitfolders
from shutil import copy, move, copyfile

import numpy as np
import pandas as pd
from PIL import Image as pil_image
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
     
from DatasetConfig_2 import config

def deleteIgnoreFile(file_list):
    '''
    移除隐文件
    '''
    for item in file_list:
        if item.startswith('.'):# os.path.isfile(os.path.join(Dogs_dir, item)):
            file_list.remove(item)
    return file_list

def makedir_help(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
def saveData(data, filename):
    joblib.dump(data, filename)

def copy_flower_2():
    '''
    把别人划分好的文件夹,整合到train_dir
    '''
    # 测试集|验证集dir
    origin_common_dir = "/data/mml/overlap_v2_datasets/flower_2/party_A/Validation Data"
    # 训练集dir
    target_common_dir = "/data/mml/overlap_v2_datasets/flower_2/party_A/Training Data"
    # 文件新名字前缀
    prefix = "val_"
    # 所有的分类
    class_list = os.listdir(origin_common_dir)
    for class_name in class_list:
        print("class_name:{}".format(class_name))
        if class_name.startswith('.'):
            continue
        # 目标文件夹
        target_dir = os.path.join(target_common_dir, class_name)
        # 原始文件夹
        origin_dir = os.path.join(origin_common_dir, class_name)
        # 原始文件夹中的文件list
        file_name_list = os.listdir(origin_dir)
        for file_name in file_name_list:
            new_file_name = prefix+file_name
            from_path = os.path.join(origin_dir, file_name)      
            to_path = os.path.join(target_dir, new_file_name)
            move(from_path, to_path)

def copy_data():
    dir_path = "/Users/mml/workspace/dataSets/overlap_datasets/sports/73_Sports_Image_Classification/dataset/valid"
    dirs = os.listdir(dir_path)
    for class_name in dirs:
        if class_name.startswith('.'):
            continue
        target_dir_path = os.path.join("/Users/mml/workspace/dataSets/overlap_datasets/sports/73_Sports_Image_Classification/dataset_pure", class_name) 
        exits_file_names = os.listdir(target_dir_path)
        exits_file_names.sort()
        index = int(exits_file_names[-1].split('.')[0])

        cur_dir = os.path.join(dir_path, class_name)
        file_names = os.listdir(cur_dir)
        file_names.sort()
        for file_name in file_names:
            from_path = os.path.join(cur_dir, file_name)
            if int(index+1) >= 100:
                new_file_name = str(int(index+1))+'.'+'jpg'
            elif int(index+1) >= 10:
                new_file_name = '0'+str(int(index+1))+'.'+'jpg'
            else:
                new_file_name = '00'+str(int(index+1))+'.'+'jpg'
            to_path = os.path.join(target_dir_path, new_file_name)
            copy(from_path, to_path)
            index = index + 1
        print(class_name,"success")

def split_data():
    # 数据划分
    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`
    origin_dir = "/data/mml/overlap_v2_datasets/flower_2/party_B/dataset_origin"
    target_dir = "/data/mml/overlap_v2_datasets/flower_2/party_B/dataset_split"
    splitfolders.ratio(origin_dir, output=target_dir, seed=1337, ratio=(.8, .2), group_prefix=None) # default values
    print("split_data() success")

def split_df(df):
    df_train,df_test = train_test_split(df,test_size = 0.5,random_state = 666, stratify=df['label'])
    return df_train, df_test

def count():
    '''
    统计该数据集各个类别样本数
    '''
    train_dir = "/Users/mml/workspace/dataSets/overlap_datasets/weather/overlap_datadatdaset/Weather_Image_Recognition"
    class_dir = os.listdir(train_dir)
    class_dir = deleteIgnoreFile(class_dir)
    class_dir.sort()
    count = 0
    for class_name in class_dir:
        cur_class_dir = os.path.join(train_dir, class_name)
        file_name_list = os.listdir(cur_class_dir)
        file_name_list = deleteIgnoreFile(file_name_list)
        file_name_list.sort()
        count += len(file_name_list)
        print("{},{}".format(class_name, len(file_name_list)))
        # print(len(file_name_list))
    print(count)

def generate_CSV(dir_path, source):
    ## 得到所有的类名
    class_name_list = os.listdir(dir_path)
    class_name_list = deleteIgnoreFile(class_name_list)
    ## 类名按照字典序（importent）
    class_name_list.sort()
    file_path_list = []
    label_name_list = []
    label_localIndex_list = []
    source_list = []
    # 遍历类名
    for index, class_name in enumerate(class_name_list):
        cur_dir = os.path.join(dir_path, class_name)
        file_name_list = os.listdir(cur_dir)
        file_name_list = deleteIgnoreFile(file_name_list)
        for file_name in file_name_list:
            file_path = os.path.join(cur_dir, file_name)
            file_path_list.append(file_path)
            label_name_list.append(class_name)
            label_localIndex_list.append(index)
            source_list.append(source)

    Fseries=pd.Series(file_path_list, name='file_path')
    Lseries=pd.Series(label_name_list, name='label')
    L_indexSeries = pd.Series(label_localIndex_list, name='label_localIndex')
    source_Series = pd.Series(source_list, name='source')
    dataset_df = pd.concat([Fseries, Lseries, L_indexSeries, source_Series], axis=1)
    return dataset_df

def setGlobal_index(data_df, dir_path_A, dir_path_B):
    class_name_list_party_A = os.listdir(dir_path_A)
    class_name_list_party_A = deleteIgnoreFile(class_name_list_party_A)
    class_name_list_party_B = os.listdir(dir_path_B)
    class_name_list_party_B = deleteIgnoreFile(class_name_list_party_B)
    global_label_name_list = list(set(class_name_list_party_A+class_name_list_party_B))
    global_label_name_list.sort() # 大写英文字母优先
    label_globalIndex_list = []
    for row_index, row in data_df.iterrows():
        label_globalIndex_list.append(global_label_name_list.index(row["label"]))
    label_globalIndex_series = pd.Series(label_globalIndex_list, name="label_globalIndex")
    data_df = pd.concat([data_df, label_globalIndex_series], axis=1)
    return data_df

def setOverlap(data_df, dir_path_A, dir_path_B):
    class_name_list_party_A = os.listdir(dir_path_A)
    class_name_list_party_A = deleteIgnoreFile(class_name_list_party_A)
    class_name_list_party_B = os.listdir(dir_path_B)
    class_name_list_party_B = deleteIgnoreFile(class_name_list_party_B)
    
    jiao_list = list(set(class_name_list_party_A) & set(class_name_list_party_B))
    is_overlap_list = []
    for row_index, row in data_df.iterrows():
        if row["label"] in jiao_list:
            is_overlap_list.append(1)
        else:
            is_overlap_list.append(0)
    is_overlap_series = pd.Series(is_overlap_list, name="is_overlap")
    data_df = pd.concat([data_df, is_overlap_series], axis=1)
    return data_df

def classToDir(df, img_dir, common_dir):
    '''
    把所有图片归类到分类文件夹
    args:
        df: dataFrame(filename, label)
        img_dir: 所有图片目录
        common_dir: 一个装数据集的目录
    '''
    for row_index, row in df.iterrows():
        class_name = row["label"]
        file_name = row["filename"]
        file_path = os.path.join(img_dir, file_name)
        target_dir = os.path.join(common_dir, class_name)
        isExit = os.path.exists(target_dir)
        if isExit is True:
            # 如果目录存在, 直接拷贝
            source = file_path
            target = os.path.join(target_dir, file_name)
            copyfile(source, target)
        else:
            # 如果目录不存在, 先创建再拷贝
            os.mkdir(target_dir)
            source = file_path
            target = os.path.join(target_dir, file_name)
            copyfile(source, target)

def classToDir_sport(df, img_dir, common_dir):
    for row_index, row in df.iterrows():
        class_name = row["labels"]
        relative_file_path = row["filepaths"]
        from_where = row["data set"]
        file_name = relative_file_path.split("/")[-1]
        abs_file_path = os.path.join(img_dir, relative_file_path)
        target_dir = os.path.join(common_dir, class_name)
        isExit = os.path.exists(target_dir)
        if isExit is True:
            # 如果目录存在, 直接拷贝
            source = abs_file_path
            target = os.path.join(target_dir, from_where+"_"+file_name)
            copyfile(source, target)
        else:
            os.mkdir(target_dir)
            source = abs_file_path
            target = os.path.join(target_dir, from_where+"_"+file_name)
            copyfile(source, target)

def eval():
    # 加载模型
    model = load_model("/Users/mml/workspace/custom_dataset/part_B/percent_20_adv/models/model_035_0.9940.h5")
    # 加载数据
    im_gen = ImageDataGenerator()
    val_dataset_gen = im_gen.flow_from_directory(
        directory = "/Users/mml/workspace/custom_dataset/part_B/percent_20_adv/dataset/train",
        # classes = ["circle", "kite", "parallelogram", "square", "trapezoid", "triangle"],
        # classes = ["circle", "kite", "parallelogram"],
        classes = ["parallelogram","square","trapezoid", "triangle"],
        target_size = (224,224),
        class_mode = "categorical",
        batch_size = 30,
        color_mode='rgb',
        shuffle=False
        )
    acc = model.evaluate(val_dataset_gen, batch_size = 30, verbose=1, return_dict=True)
    print(acc)
    # print(model.predict(val_dataset_gen, batch_size=None, verbose=0, steps=None))
    # x = np.array(Image.open("/Users/mml/workspace/dataSets/overlap_datasets/shapes/Geometric_Shapes_Mathematics/six-shapes-dataset-v1/dataset_artificial_pivot/part_2/preservation/val/model_part_1/parallelogram_adv/parallelogram-test-103.png").convert('RGB').resize((224,224))) 
    # x = x.reshape(-1, 224, 224, 3)
    # print(model.predict(x))

def modifyCSV_file_path(df):
    '''
    修改csv 中file_path字段到相对路经
    '''
    new_file_path_list = []
    for row_index, row in df.iterrows():
        old_path = row["file_path"]
        cur_path =  old_path.split("overlap_v2_datasets/")[1]
        new_file_path_list.append(cur_path)
    df["file_path"] = new_file_path_list
    print("modifyCSV_file_path() success")
    return df

def getLocalToGlobal(party_A_classDir, party_B_classDir):
    class_name_list_party_A = os.listdir(party_A_classDir)
    class_name_list_party_A = deleteIgnoreFile(class_name_list_party_A)
    class_name_list_party_A.sort() # 局部类名按照字典序
    class_name_list_party_B = os.listdir(party_B_classDir)
    class_name_list_party_B = deleteIgnoreFile(class_name_list_party_B)
    class_name_list_party_B.sort() # 局部类名按照字典序
    global_label_name_list = list(set(class_name_list_party_A+class_name_list_party_B))
    global_label_name_list.sort() # 大写英文字母优先
    local_to_gobal_party_A = {}
    for localIndex, labelName in enumerate(class_name_list_party_A):
        local_to_gobal_party_A[localIndex] = global_label_name_list.index(labelName)
    
    local_to_gobal_party_B = {}
    for localIndex, labelName in enumerate(class_name_list_party_B):
        local_to_gobal_party_B[localIndex] = global_label_name_list.index(labelName)

    
    print("getLocalToGlobal() success")
    return local_to_gobal_party_A, local_to_gobal_party_B

def concat(csv_1_filePath, csv_2_filePath):
    '''
    将两个dataFrame 上下链接起来
    '''
    df_1 = pd.read_csv(csv_1_filePath)
    df_2 = pd.read_csv(csv_2_filePath)

    merged_df = pd.concat([df_1, df_2], ignore_index=True)
    # merged_df.rename(columns={'Unnamed: 0':'idx'}, inplace=True)
    # print(merged_df["Unnamed: 0"])
    # print(merged_df.iloc[5992:])
    # save_path = os.path.join(save_dir, save_fileName)
    # merged_df.to_csv(save_path) # index=False
    # merged_df.to_csv('./saved/shapes/val/merged_df.csv', index=False)
    # merged_df.to_csv('./saved/weather/val/merged_df.csv', index=False)
    print("concat success")
    return merged_df

def getOverlap_df(csv_path):
    '''
    从dataFrame 中 抽出 overlap
    '''
    df = pd.read_csv(csv_path)
    overlap_df = df[df["is_overlap"] == 1]
    print("getOverlap_df() success")
    return overlap_df

def getUnique_df(csv_path):
    '''
    从dataFrame 中 抽出 unique
    '''
    df = pd.read_csv(csv_path)
    unique_df = df[df["is_overlap"] == 0]
    print("getUnique_df() success")
    return unique_df

def getOverlapGlobalLabelIndex(local_to_gobal_party_A, local_to_gobal_party_B):
    globalLabelIndexValueList_A = local_to_gobal_party_A.values()
    globalLabelIndexValueList_B = local_to_gobal_party_B.values()
    overlapGlobalLabelIndex_list = list(set(globalLabelIndexValueList_A).intersection(set(globalLabelIndexValueList_B)))
    overlapGlobalLabelIndex_list.sort()
    return overlapGlobalLabelIndex_list

def deleteInvalidImage():
    common_dir = "/data/mml/overlap_v2_datasets/animal_2/party_A/data_origin"
    count_noExist = 0
    count_hidden = 0
    count_NoOpen = 0
    count_txt =0
    total = 0
    classDir_list = os.listdir(common_dir)
    for classDir in classDir_list:
        print("classDir:{}".format(classDir))
        if classDir.startswith("."):
            os.remove(os.path.join(common_dir, classDir))
            continue
        curDir = os.path.join(common_dir, classDir)
        file_name_list = os.listdir(curDir)
        for file_name in file_name_list:
            total += 1
            # 得到文件路径
            file_path = os.path.join(curDir, file_name)
            # 删除不存在的图片
            if not os.path.exists(file_path):
                # os.remove(file_path)
                count_noExist += 1
                continue
            # 删除隐藏文件
            if file_name.startswith('.'):
                print(file_path)
                # os.remove(file_path)
                count_hidden += 1
                continue
            # 删除图片文件配套的txt说明文件
            if file_name.split(".")[-1] == "txt":
                # os.remove(file_path)
                count_txt += 1
                continue
            # 删除pil_image打不开的
            if not is_read_successfully(file_path):
                # print(file_path)
                # os.remove(file_path)
                # print(file_path)
                count_NoOpen += 1
                continue
            # try:
            #     # with open(file_path, 'rb') as f:
            #     #     img = pil_image.open(io.BytesIO(f.read()))
            # except Exception:
            #     os.remove(file_path)
            #     count += 1
            #     continue
    print("共,{}个文件".format(total))    
    print("删除不存在文件, {}个".format(count_noExist))
    print("删除隐藏文件, {}个".format(count_hidden))
    print("删除打开失败文件, {}个".format(count_NoOpen))
    print("删除txt文件, {}个".format(count_txt))
    print("共删除,{}个文件".format(count_noExist+count_hidden+count_NoOpen+count_txt))           
    
def is_read_successfully(file):
    try:
        imgFile = pil_image.open(file) #这个就是一个简单的打开成功与否
        with open(file, 'rb') as f:
            img = pil_image.open(io.BytesIO(f.read()))
        return True
    except Exception:
        return False

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

if __name__ == "__main__":
    # copy_flower_2()
    '''
    删除数据集中无效文件
    '''
    # deleteInvalidImage()
    '''
    切分数据集
    '''
    # split_data()
    '''
    统计各个类别样本数
    '''
    # count()
   
    '''
    形成csv
    '''
    # dir_path = "/data/mml/overlap_v2_datasets/flower_2/party_B/dataset_split/val"
    # source = 2 # importent!!!
    # dataset_df = generate_CSV(dir_path, source)
    # print(dataset_df.head())
    # saved_dir = "/data/mml/overlap_v2_datasets/flower_2/party_B/dataset_split"
    # dataset_df.to_csv(os.path.join(saved_dir, "val.csv"), index=True, index_label="logic_id")

    '''
    setGlobal_index
    '''
    # df = pd.read_csv('/data/mml/overlap_v2_datasets/flower_2/party_B/dataset_split/val.csv')
    # dir_path_A = "/data/mml/overlap_v2_datasets/flower_2/party_A/dataset_split/val"
    # dir_path_B = "/data/mml/overlap_v2_datasets/flower_2/party_B/dataset_split/val"
    # df_withG = setGlobal_index(df, dir_path_A, dir_path_B)
    # print(df_withG.head())
    # saved_dir = "/data/mml/overlap_v2_datasets/flower_2/party_B/dataset_split"
    # df_withG.to_csv(os.path.join(saved_dir, "val_withG.csv"), index=False)

    '''
    setOverlap
    '''
    # data_df = pd.read_csv('/data/mml/overlap_v2_datasets/flower_2/party_B/dataset_split/val_withG.csv')
    # dir_path_A = "/data/mml/overlap_v2_datasets/flower_2/party_A/dataset_split/val"
    # dir_path_B = "/data/mml/overlap_v2_datasets/flower_2/party_B/dataset_split/val"
    # df_withG_Overlap = setOverlap(data_df, dir_path_A, dir_path_B)
    # print(df_withG_Overlap.head())
    # saved_dir = "/data/mml/overlap_v2_datasets/flower_2/party_B/dataset_split"
    # df_withG_Overlap.to_csv(os.path.join(saved_dir, "val_withG_Overlap.csv"), index=False)


    '''
    modifyCSV_file_path
    '''
    # df = pd.read_csv("/data/mml/overlap_v2_datasets/flower_2/party_B/dataset_split/val_withG_Overlap.csv")
    # df = modifyCSV_file_path(df)
    # saved_dir = "/data/mml/overlap_v2_datasets/flower_2/party_B/dataset_split"
    # df.to_csv(os.path.join(saved_dir, "val_relative_path.csv"), index=False)


    '''
    get local_to_global
    '''
    # party_A_classDir = "/data/mml/overlap_v2_datasets/flower_2/party_A/dataset_split/train"
    # party_B_classDir = "/data/mml/overlap_v2_datasets/flower_2/party_B/dataset_split/train"
    # local_to_gobal_party_A, local_to_gobal_party_B = getLocalToGlobal(party_A_classDir, party_B_classDir)
    # save_dir = "exp_data/flower_2/LocalToGlobal"
    # saveData(local_to_gobal_party_A, os.path.join(save_dir, "local_to_gobal_party_A.data")) 
    # saveData(local_to_gobal_party_B, os.path.join(save_dir, "local_to_gobal_party_B.data")) 
    # local_to_gobal_party_A = joblib.load(os.path.join(save_dir, "local_to_gobal_party_A.data"))
    # local_to_gobal_party_B = joblib.load(os.path.join(save_dir, "local_to_gobal_party_B.data"))
    # print("success()")

    
    '''
    上下拼接df
    '''
    # csv_1_filePath = "/data/mml/overlap_v2_datasets/flower_2/party_A/dataset_split/val.csv"
    # csv_2_filePath = "/data/mml/overlap_v2_datasets/flower_2/party_B/dataset_split/val.csv"
    # merged_df = concat(csv_1_filePath, csv_2_filePath)
    # save_dir = "/data/mml/overlap_v2_datasets/flower_2/merged_data/test"
    # save_fileName = "merged_df.csv"
    # merged_df.to_csv(os.path.join(save_dir, save_fileName), index=True, index_label="merged_idx")

    '''
    从df 抽取 overlap | Unique_df
    '''
    # csv_path = "/data/mml/overlap_v2_datasets/flower_2/merged_data/test/merged_df.csv"
    # overlap_df = getUnique_df(csv_path) #  getOverlap_df(csv_path) | getUnique_df(csv_path)
    # save_dir = "/data/mml/overlap_v2_datasets/flower_2/merged_data/test"
    # file_name = "merged_df_unique.csv"
    # overlap_df.to_csv(os.path.join(save_dir, file_name), index=False)

    '''
    把图片归类到文件夹中
    '''
    # df = pd.read_csv("/Users/mml/workspace/dataSets/overlap_v2_datasets/food/party_B/Training_set_food.csv")
    # img_dir = "/Users/mml/workspace/dataSets/overlap_v2_datasets/food/party_B/train/train"
    # common_dir = "/Users/mml/workspace/dataSets/overlap_v2_datasets/food/party_B/dataset"
    # classToDir(df, img_dir, common_dir)

    '''
    把图片归类到文件夹中(sport)
    '''
    # df = pd.read_csv("/Users/mml/workspace/dataSets/overlap_v2_datasets/sport/party_A/sports.csv")
    # img_dir = "/Users/mml/workspace/dataSets/overlap_v2_datasets/sport/party_A"
    # common_dir = "/Users/mml/workspace/dataSets/overlap_v2_datasets/sport/party_A/dataset"
    # classToDir_sport(df, img_dir, common_dir)


    '''
    双方模型评估
    '''
    # eval()

    '''
    划分merged_test_dataset
    '''
    # dataset_name = config['dataset_name']
    # root_dir = config["root_dir"]
    # dataset_csv_path = os.path.join(root_dir,dataset_name,"merged_data","test","merged_df.csv")
    # df = pd.read_csv(dataset_csv_path)
    # df_train, df_test = split_df(df)
    # save_dir = f"exp_data/{dataset_name}/sampling/percent/random_split/test"
    # makedir_help(save_dir)
    # save_file_name = "test.csv"
    # save_file_path = os.path.join(save_dir, save_file_name)
    # df_test.to_csv(save_file_path, index=False)
    # total_num = df.shape[0]
    # sample_rate_list = config["sample_rate_list"]
    # for sample_rate in sample_rate_list:
    #     sample_num = math.ceil(total_num*sample_rate)
    #     for i in range(10):
    #         sampled_df = df_train.sample(n=sample_num, axis=0) # random_state=123    
    #         save_dir = f"exp_data/{dataset_name}/sampling/percent/random_split/train/{int(sample_rate*100)}" 
    #         makedir_help(save_dir)
    #         save_file_name =f"sample_{i}.csv" 
    #         save_file_path = os.path.join(save_dir, save_file_name)
    #         sampled_df.to_csv(save_file_path, index=False)
    # pass
    pass