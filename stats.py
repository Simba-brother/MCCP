from tensorflow.keras.models import Model, Sequential, load_model
import pandas as pd
import numpy as np
import joblib

from utils import deleteIgnoreFile, saveData, makedir_help

import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from DataSetConfig import food_config, fruit_config, sport_config, weather_config, flower_2_config, car_body_style_config, animal_config, animal_2_config, animal_3_config
import Base_acc
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras import optimizers


def generate_generator_multiple(batches_A, batches_B):
    '''
    将连个模型的输入bath 同时返回
    '''
    while True:
        X1i = batches_A.next()
        X2i = batches_B.next()
        yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label

def eval_combination_model(model, df, generator_left, generator_right, classes, target_size_A, target_size_B):
    y_col = "label"
    batch_size = 32
    batches_A = generator_left.flow_from_dataframe(df, 
                                                    directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                    # subset="training",
                                                    seed=42,
                                                    x_col='file_path', y_col=y_col, 
                                                    target_size=target_size_A, class_mode='categorical',
                                                    color_mode='rgb', classes=classes, shuffle=False, batch_size=batch_size,
                                                    validate_filenames=False)
    batches_B = generator_right.flow_from_dataframe(df, 
                                                    directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                    # subset="training",
                                                    seed=42,
                                                    x_col='file_path', y_col=y_col, 
                                                    target_size=target_size_B, class_mode='categorical',
                                                    color_mode='rgb', classes=classes, shuffle=False, batch_size=batch_size,
                                                    validate_filenames=False)

    batches = generate_generator_multiple(batches_A, batches_B)
    batch_size = 32
    eval_matric = model.evaluate(batches, batch_size = batch_size, verbose=1,steps = batches_A.n/batch_size, return_dict=True)
    return eval_matric["accuracy"]

def add_reserve_class(model):
    '''
    给模型增加一个保留class神经元
    '''
    # copy the original weights, to keep the predicting function same
    weights_bak = model.layers[-1].get_weights()
    num_classes = model.layers[-1].output_shape[-1]
    # model.pop()
    # model.add(Dense(num_classes + 1, activation='softmax'))
    # cut 最后输出
    model_cut = keras.Model(inputs = model.input, outputs = model.get_layer(index = -2).output)
    # 声明出一个新输出层
    new_output_layer = Dense(num_classes + 1, activation='softmax', name="new_output_layer")(model_cut.output)
    # 重新连上新输出层
    model = Model(inputs = model_cut.input, outputs = new_output_layer)
    # model.summary()
    weights_new = model.layers[-1].get_weights()
    weights_new[0][:, :-1] = weights_bak[0]
    weights_new[1][:-1] = weights_bak[1]

    # use the average weight to init the last. This suppress its output, while keeping performance.
    weights_new[0][:, -1] = np.mean(weights_bak[0], axis=1)
    weights_new[1][-1] = np.mean(weights_bak[1])

    model.layers[-1].set_weights(weights_new)

    # model.compile(loss=categorical_crossentropy,
    #             optimizer=Adam(learning_rate=1e-4),
    #             metrics=['accuracy'])
    return model

def getClasses(dir_path):
    '''
    得到数据集目录的class_name_list
    '''
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    classes_name_list.sort()
    return classes_name_list

def get_pLabel_model(model, df, generator, target_size, local_to_global):
    global_pLabels= []
    local_pLabels = []
    batch_size = 5
    batches = generator.flow_from_dataframe(df, 
                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                            x_col='file_path', y_col="label", 
                            target_size=target_size, class_mode='categorical', # one-hot
                            color_mode='rgb', classes=None,
                            shuffle=False, batch_size=batch_size,
                            validate_filenames=False)
    probs = model.predict_generator(batches, steps=batches.n/batch_size)
    confidences = np.max(probs,axis=1)
    pseudo_label_indexes = np.argmax(probs, axis=1)
    for i in range(pseudo_label_indexes.shape[0]):
        local_pLabel = pseudo_label_indexes[i]
        global_pLabel = local_to_global[local_pLabel]
        global_pLabels.append(global_pLabel)
        local_pLabels.append(local_pLabel)
    global_pLabels = np.array(global_pLabels)
    return global_pLabels, local_pLabels, confidences

def local_to_global(localToGlobal_mapping,proba_local):
    '''
    i方的proba => globa proba
    '''
    predict_value = np.zeros((proba_local.shape[0], all_classes_num))
    localToGlobal_dic = localToGlobal_mapping
    mapping = []
    for key, value in localToGlobal_dic.items():
        mapping.append(value)
    predict_value[:, mapping] = proba_local
    return predict_value

def get_probs_extendedModel(model, df, generator, target_size):
    batch_size = 5
    batches = generator.flow_from_dataframe(df, 
                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                            x_col='file_path', y_col="label", 
                            target_size=target_size, class_mode='categorical', # one-hot
                            color_mode='rgb', classes=None,
                            shuffle=False, batch_size=batch_size,
                            validate_filenames=False)
    probs = model.predict_generator(batches, steps=batches.n/batch_size)

    return probs[:,:-1]

def get_ourCombin_model_pLabel(config):
    combin_model = load_model(config["combination_model_path"])
    prefix_path = "/data/mml/overlap_v2_datasets/"
    batch_size = 5
    test_batches_A = generator_A_test.flow_from_dataframe(merged_test_df, 
                                            directory = prefix_path, # 添加绝对路径前缀
                                            x_col='file_path', y_col='label', 
                                            target_size=target_size_A, class_mode='categorical',
                                            color_mode='rgb', classes = None, shuffle=False, batch_size=batch_size,
                                            validate_filenames=False)
                                                                                                                # weather:rgb  150, 150

    test_batches_B = generator_B_test.flow_from_dataframe(merged_test_df, 
                                                directory = prefix_path, # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size_B, class_mode='categorical',
                                                color_mode='rgb', classes = None, shuffle=False, batch_size=batch_size,
                                                validate_filenames=False)

    test_batches = get_combin_batches(test_batches_A, test_batches_B)
    probs =combin_model.predict(test_batches, batch_size = batch_size, verbose=0,steps = test_batches_A.n/batch_size)
    p_labels = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    return p_labels, confidences

def get_combin_batches(batches_A, batches_B):

    while True:
        X1i = batches_A.next()
        X2i = batches_B.next()
        yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label

def write(p_Label_A, p_Label_B, p_Label_C, ground_truth, confidences, row):

    row["ALL"]["num"] += 1
    p_Label = None
    if p_Label_A == p_Label_B and p_Label_A == p_Label_C:
        row["ABC"]["num"] += 1
        p_Label = p_Label_C
        if ground_truth == p_Label:
            row["ABC"]["correct_num"] += 1
            row["ALL"]["correct_num"] += 1
    elif p_Label_C == p_Label_A and p_Label_C != p_Label_B:
        row["CA"]["num"] += 1
        p_Label = p_Label_C
        if ground_truth == p_Label:
            row["CA"]["correct_num"] += 1
            row["ALL"]["correct_num"] += 1
    elif p_Label_C == p_Label_B and p_Label_C != p_Label_A:
        row["CB"]["num"] += 1
        p_Label = p_Label_C
        if ground_truth == p_Label:
            row["CB"]["correct_num"] += 1
            row["ALL"]["correct_num"] += 1
    elif p_Label_A == p_Label_B and p_Label_A != p_Label_C:
        row["AB"]["num"] += 1
        p_Label = p_Label_A
        if ground_truth == p_Label:
            row["AB"]["correct_num"] += 1
            row["ALL"]["correct_num"] += 1
    else:
        row["UNIQUE"]["num"] += 1
        p_Labels = [p_Label_A, p_Label_B, p_Label_C]
        p_Label = p_Labels[np.argmax(confidences)]
        if ground_truth == p_Label:
            row["UNIQUE"]["correct_num"] += 1
            row["ALL"]["correct_num"] += 1
    return row

def get_dummy_row():
    row = {
        "ABC":{"num":0, "correct_num":0, "percent":0}, 
        "CA":{"num":0, "correct_num":0, "percent":0}, 
        "CB":{"num":0, "correct_num":0, "percent":0}, 
        "AB":{"num":0, "correct_num":0, "percent":0}, 
        "UNIQUE":{"num":0,"correct_num":0, "percent":0}, 
        "ALL":{"num":0, "correct_num":0, "percent":0}
        }
    global_pLabels_A, local_pLabels_A, confidences_A = get_pLabel_model(model_A, merged_test_df, generator_A_test, target_size_A, local_to_global_party_A)
    global_pLabels_B, local_pLabels_B, confidences_B = get_pLabel_model(model_B, merged_test_df, generator_B_test, target_size_B, local_to_global_party_B)
    
    ground_truths = list(merged_test_df["label_globalIndex"])
    for i in range(global_pLabels_A.shape[0]):
        global_pLabel_A = global_pLabels_A[i]
        global_pLabel_B = global_pLabels_B[i]
        ground_truth = ground_truths[i]

        confidence_A = confidences_A[i]        
        confidence_B = confidences_B[i]
        if confidence_A > confidence_B:
            global_pLabel_Dummy = global_pLabel_A
            confidence_Dummy = confidence_A
        else:
            global_pLabel_Dummy = global_pLabel_B
            confidence_Dummy = confidence_B
        confidences = [confidence_A, confidence_B, confidence_Dummy]
        row = write(global_pLabel_A, global_pLabel_B, global_pLabel_Dummy, ground_truth, confidences, row)
    for col_name in ["ABC", "CA", "CB", "AB", "UNIQUE", "ALL"]:
        num = row[col_name]["num"]
        correct_num = row[col_name]["correct_num"]
        if num == 0:
            acc = 0.0000
        else:
            acc = round(correct_num/num, 4)
        row[col_name]["acc"] = acc
    return row



def get_hmr_row(config):
    row = {
            "ABC":{"num":0, "correct_num":0, "percent":0}, 
            "CA":{"num":0, "correct_num":0, "percent":0}, 
            "CB":{"num":0, "correct_num":0, "percent":0}, 
            "AB":{"num":0, "correct_num":0, "percent":0}, 
            "UNIQUE":{"num":0,"correct_num":0, "percent":0}, 
            "ALL":{"num":0, "correct_num":0, "percent":0}
        }
    
    model_a = load_model(config["model_A_struct_path"])
    model_a.load_weights(config["model_A_weight_path"])
    model_b = load_model(config["model_B_struct_path"])
    if not config["model_B_weight_path"] is None:
        model_b.load_weights(config["model_B_weight_path"])

    global_pLabels_A, local_pLabels_A, confidences_A = get_pLabel_model(model_a, merged_test_df, generator_A_test, target_size_A, local_to_global_party_A)
    global_pLabels_B, local_pLabels_B, confidences_B = get_pLabel_model(model_b, merged_test_df, generator_B_test, target_size_B, local_to_global_party_B)
    ground_truths = list(merged_test_df["label_globalIndex"])
    model_a = add_reserve_class(model_a)
    model_b = add_reserve_class(model_b)
    n= merged_test_df.shape[0]
    predict_value = np.zeros((2, n, all_classes_num))
    proba_local_A = get_probs_extendedModel(model_a, merged_test_df, generator_A_test, target_size_A)
    predict_value[0,:,:] = local_to_global(local_to_global_party_A, proba_local_A)
    proba_local_B = get_probs_extendedModel(model_b, merged_test_df, generator_B_test, target_size_B)
    predict_value[1,:,:] = local_to_global(local_to_global_party_B, proba_local_B)
    proba_vector = np.max(predict_value, axis=0)
    global_pLabels_HMR = np.argmax(proba_vector,axis=1)
    confidences_HMR = np.max(proba_vector, axis = 1)

    for i in range(global_pLabels_A.shape[0]):
        global_pLabel_A = global_pLabels_A[i]
        global_pLabel_B = global_pLabels_B[i]
        global_pLabel_HMR = global_pLabels_HMR[i]
        ground_truth = ground_truths[i]

        confidence_A = confidences_A[i]        
        confidence_B = confidences_B[i]
        confidence_HMR = confidences_HMR[i]
        confidences = [confidence_A, confidence_B, confidence_HMR]
        row = write(global_pLabel_A, global_pLabel_B, global_pLabel_HMR, ground_truth, confidences, row)

    for col_name in ["ABC", "CA", "CB", "AB", "UNIQUE", "ALL"]:
        num = row[col_name]["num"]
        correct_num = row[col_name]["correct_num"]
        if num == 0:
            acc = 0.0000
        else:
            acc = round(correct_num/num, 4)
        row[col_name]["acc"] = acc
    return row

def get_CFL_row(config):
    row = {
            "ABC":{"num":0, "correct_num":0, "percent":0}, 
            "CA":{"num":0, "correct_num":0, "percent":0}, 
            "CB":{"num":0, "correct_num":0, "percent":0}, 
            "AB":{"num":0, "correct_num":0, "percent":0}, 
            "UNIQUE":{"num":0,"correct_num":0, "percent":0}, 
            "ALL":{"num":0, "correct_num":0, "percent":0}
        }
    global_pLabels_A, local_pLabels_A, confidences_A = get_pLabel_model(model_A, merged_test_df, generator_A_test, target_size_A, local_to_global_party_A)
    global_pLabels_B, local_pLabels_B, confidences_B = get_pLabel_model(model_B, merged_test_df, generator_B_test, target_size_B, local_to_global_party_B)
    ground_truths = list(merged_test_df["label_globalIndex"])

    stu_model = load_model(config["stu_model_path"])
    batch_size = 5
    generator = ImageDataGenerator(rescale=1/255.)
    batches = generator.flow_from_dataframe(merged_test_df, 
                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                            x_col='file_path', y_col="label", 
                            target_size=(224,224), class_mode='categorical', # one-hot
                            color_mode='rgb', classes=None,
                            shuffle=False, batch_size=batch_size,
                            validate_filenames=False)
    probs = stu_model.predict_generator(batches, steps=batches.n/batch_size)
    confidences_STU = np.max(probs,axis=1)
    pLabels_STU = np.argmax(probs, axis=1)

    for i in range(global_pLabels_A.shape[0]):
        global_pLabel_A = global_pLabels_A[i]
        global_pLabel_B = global_pLabels_B[i]
        gloabl_pLabel_STU = pLabels_STU[i]
        ground_truth = ground_truths[i]

        confidence_A = confidences_A[i]        
        confidence_B = confidences_B[i]
        confidence_STU = confidences_STU[i]
        confidences = [confidence_A, confidence_B, confidence_STU]
        row = write(global_pLabel_A, global_pLabel_B, gloabl_pLabel_STU, ground_truth, confidences, row)

    for col_name in ["ABC", "CA", "CB", "AB", "UNIQUE", "ALL"]:
        num = row[col_name]["num"]
        correct_num = row[col_name]["correct_num"]
        if num == 0:
            acc = 0.0000
        else:
            acc = round(correct_num/num, 4)
        row[col_name]["acc"] = acc
    return row

def get_OurCombin_row(config):
    pLabels_OurCombin, confidences_OurCombin = get_ourCombin_model_pLabel(config)
    row = {
        "ABC":{"num":0, "correct_num":0, "percent":0}, 
        "CA":{"num":0, "correct_num":0, "percent":0}, 
        "CB":{"num":0, "correct_num":0, "percent":0}, 
        "AB":{"num":0, "correct_num":0, "percent":0}, 
        "UNIQUE":{"num":0,"correct_num":0, "percent":0}, 
        "ALL":{"num":0, "correct_num":0, "percent":0}
        }
    
    global_pLabels_A, local_pLabels_A, confidences_A = get_pLabel_model(model_A, merged_test_df, generator_A_test, target_size_A, local_to_global_party_A)
    global_pLabels_B, local_pLabels_B, confidences_B = get_pLabel_model(model_B, merged_test_df, generator_B_test, target_size_B, local_to_global_party_B)
    ground_truths = list(merged_test_df["label_globalIndex"])

    for i in range(global_pLabels_A.shape[0]):
        global_pLabel_A = global_pLabels_A[i]
        global_pLabel_B = global_pLabels_B[i]
        gloabl_pLabel_OurCombin = pLabels_OurCombin[i]
        ground_truth = ground_truths[i]

        confidence_A = confidences_A[i]        
        confidence_B = confidences_B[i]
        confidence_OurCombin = confidences_OurCombin[i]
        confidences = [confidence_A, confidence_B, confidence_OurCombin]
        row = write(global_pLabel_A, global_pLabel_B, gloabl_pLabel_OurCombin, ground_truth, confidences, row)

    for col_name in ["ABC", "CA", "CB", "AB", "UNIQUE", "ALL"]:
        num = row[col_name]["num"]
        correct_num = row[col_name]["correct_num"]
        if num == 0:
            acc = 0.0000
        else:
            acc = round(correct_num/num, 4)
        row[col_name]["acc"] = acc
    return row

def get_tabel_row(data):
    row = []
    for key in ["ABC",  "CA",   "CB",  "AB",  "UNIQUE",  "ALL"]:
        num = int(data[key]['num'])
        acc = round(data[key]["acc"]*100,4)
        acc_str = str(acc)+"%"
        row.append(num)
        row.append(acc_str)
    return row

def get_table():
    df = pd.DataFrame(np.zeros(shape=(4,12)),index=["Dummy", "HMR", "CFL", "OurCombin"], 
                      columns=[["ABC", "ABC", "CA", "CA",  "CB", "CB", "AB", "AB", "UNIQUE", "UNIQUE", "ALL", "ALL"], 
                                ["num", "acc", "num", "acc","num", "acc","num", "acc","num", "acc","num", "acc"]])
    dummy_row = get_dummy_row()
    dummy_row = get_tabel_row(dummy_row)

    hmr_row = get_hmr_row(config)
    hmr_row = get_tabel_row(hmr_row)

    cfl_row = get_CFL_row(config)
    cfl_row = get_tabel_row(cfl_row)

    ourCombin_row = get_OurCombin_row(config)
    ourCombin_row = get_tabel_row(ourCombin_row)
    
    df.loc["Dummy"] = dummy_row
    df.loc["HMR"] = hmr_row
    df.loc["CFL"] = cfl_row
    df.loc["OurCombin"] = ourCombin_row
    return df

def get_Train_Test_Size():
    df_train_party_A = pd.read_csv(config["df_train_party_A_path"])
    df_train_party_B = pd.read_csv(config["df_train_party_B_path"])
    df_eval_party_A = pd.read_csv(config["df_eval_party_A_path"])
    df_eval_party_B = pd.read_csv(config["df_eval_party_B_path"])
    print(f"A:TrainSize:{df_train_party_A.shape[0]}")
    print(f"A:TestSize:{df_eval_party_A.shape[0]}")
    print(f"B:TrainSize:{df_train_party_B.shape[0]}")
    print(f"B:TrainSize:{df_eval_party_B.shape[0]}")

def get_eval_data():
    ans = {}
    model = load_model(config["combination_model_path"])
    adam = optimizers.Adam(learning_rate=config["combiantion_lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")
    df = merged_test_df
    percent_list = [1,3,5,10,15,20]
    repeat_list = [0,1,2,3,4]
    prefix_dir = f"/data/mml/overlap_v2_datasets/{dataset_name}/merged_model/trained_model_weights/combined_model"
    init_acc = eval_combination_model(model, df, generator_A_test, generator_B_test, all_classes_list, target_size_A, target_size_B)
    ans["init_acc"] = init_acc
    ans["train"] = {}
    for percent in percent_list:
        ans["train"][percent] = []
        for repeat in repeat_list:
            file_name = f"weights_{repeat}.h5"
            weights_path = os.path.join(prefix_dir, str(percent), file_name)
            model.load_weights(weights_path)
            acc = eval_combination_model(model, df, generator_A_test, generator_B_test, all_classes_list, target_size_A, target_size_B)
            ans["train"][percent].append(acc)
    return ans
#  全局变量区域
os.environ['CUDA_VISIBLE_DEVICES']='2'
config = animal_3_config
Base_acc_config = Base_acc.animal_3
dataset_name = config["dataset_name"]
model_A = load_model(config["model_A_struct_path"])
model_A.load_weights(config["model_A_weight_path"])
model_B = load_model(config["model_B_struct_path"])
if not config["model_B_weight_path"] is None:
    model_B.load_weights(config["model_B_weight_path"])
merged_test_df = pd.read_csv(config["merged_df_path"])

generator_A_test = config["generator_A_test"]
generator_B_test = config["generator_B_test"]
target_size_A = config["target_size_A"]
target_size_B = config["target_size_B"]
local_to_global_party_A = joblib.load(config["local_to_global_party_A_path"])
local_to_global_party_B = joblib.load(config["local_to_global_party_B_path"])
# 双方的class_name_list
classes_A = getClasses(config["dataset_A_train_path"]) # sorted
classes_B = getClasses(config["dataset_B_train_path"]) # sorted
# all_class_name_list
all_classes_list = list(set(classes_A+classes_B))
all_classes_list.sort()
# 总分类数
all_classes_num = len(all_classes_list)


def test():
    pass

if __name__ == "__main__":
    
    # train_acc = get_eval_data()
    # save_dir = f"exp_data/{dataset_name}/retrainResult/percent/OurCombin"
    # file_name = "train_acc_v3.data"
    # saveData(train_acc, os.path.join(save_dir, file_name))

    # get_Train_Test_Size()
    # df = get_table()
    # save_dir = f"exp_table/{dataset_name}"
    # file_name = "vote.xlsx"
    # file_path = os.path.join(save_dir, file_name)
    # df.to_excel(file_path)
    pass