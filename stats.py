import sys
import os
from tensorflow.keras.models import Model, Sequential, load_model
import pandas as pd
import numpy as np
import joblib
from utils import deleteIgnoreFile, saveData, makedir_help
from scipy.stats import spearmanr
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from DataSetConfig import food_config, fruit_config, sport_config, weather_config, flower_2_config, car_body_style_config, animal_config, animal_2_config, animal_3_config
import Base_acc
from model_eval import eval_combination_Model
from HMR import load_models_pool, evaluate_on
from CFL_simple import eval_stu_model
from dummy import dummy_eval

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
    classes_name_list.sort() # 重要！！！
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
    df = pd.read_csv(config["merged_overlap_df"])
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

def get_table_causal_variables():
    def get_row(dataset_name):
        row = []
        # 加载模型
        model = load_model(f"/data/mml/overlap_v2_datasets/{dataset_name}/merged_model/combination_model_inheritWeights.h5")
        initAcc_dic = joblib.load(f"exp_data/{dataset_name}/initAcc.data")
        init_acc = initAcc_dic["all_initAcc"]["accuracy"]
        merged_train_df = pd.read_csv(f"/data/mml/overlap_v2_datasets/{dataset_name}/merged_data/train/merged_df.csv")
        classes = merged_train_df["label"].unique()
        classes = np.sort(classes).tolist()
        categories = len(classes)
        df = pd.read_csv(f"/data/mml/overlap_v2_datasets/{dataset_name}/merged_data/test/merged_df.csv")
        dataset_size = df.shape[0]
        trainable_count = int(np.sum([K.count_params(p) for p in model.trainable_weights]))
        non_trainable_count = int(np.sum([K.count_params(p) for p in model.non_trainable_weights]))
        model_param_size = trainable_count + non_trainable_count
        train_acc = joblib.load(f"exp_data/{dataset_name}/retrainResult/percent/OurCombin/train_acc_v3.data")
        avg = get_avg(train_acc)
        row.append(model_param_size)
        row.append(dataset_size)
        row.append(categories)
        row.append(init_acc)
        row.extend(avg)
        return row
    def get_avg(train_acc):
        ans = []
        percent_list = [1,3,5,10,15,20]
        for percent in percent_list:
            ans.append(np.mean((train_acc["train"][percent])))
        return ans
    index = ["car", "flower", "food", "fruit", "sport", "weather", "animal_1", "animal_2", "animal_3"]
    columns = ["model_param_size", "merged_test_dataset_size", "categories", "init_acc", "1%", "3%", "5%", "10%", "15%", "20%"]
    # car
    row_car = get_row("car_body_style")
    # flower
    row_flower = get_row("flower_2")
    # food
    row_food = get_row("food")
    # fruit
    row_fruit = get_row("Fruit")
    # sport
    row_sport = get_row("sport")
    # weather 
    row_weather = get_row("weather")
    # animal
    row_animal = get_row("animal")
    # animal_2
    row_animal_2 = get_row("animal_2")
    # animal_3
    row_animal_3 = get_row("animal_3")
    data = [row_car, row_flower, row_food, row_fruit, row_sport, row_weather, row_animal, row_animal_2, row_animal_3]
    df = pd.DataFrame(data, index, columns)
    return df

def calc_correlation():
    df = pd.read_csv("exp_data/all/causal_variables.csv", index_col=[0])    
    corr = df.corr(method='spearman', min_periods=1)
    new_corr = corr.iloc[0:4,4:10]
    save_dir = "exp_data/all"
    file_name = "spearman_corr.csv"
    file_path = os.path.join(save_dir, file_name)
    new_corr.to_csv(file_path)
    print("calc_correlation successfully!")

def get_init_eval(config):
    all_df = pd.read_csv(config["merged_df_path"])
    overlap_df = pd.read_csv(config["merged_overlap_df"])
    unique_df = pd.read_csv(config["merged_unique_df"])

    eval_df = unique_df
    # ==== our_combination ====
    res = eval_combination_Model(config,eval_df)
    acc_our = round(res["accuracy"],4)
    # ==== HMR ====
    models_pool,_ = load_models_pool(config, lr=1e-3)
    acc_hmr = evaluate_on(models_pool, eval_df)  # 要去换方法所在文件的config

    # ==== CFL ====
    model = load_model(config['stu_model_path'])
    generator = ImageDataGenerator(rescale=1/255.)
    acc_cfl = eval_stu_model(model, eval_df, generator, (224,224), all_classes_list, lr=1e-3)
    # ==== Dummy ====
    acc_dummy = dummy_eval(config, eval_df)
    print(f"our: {acc_our} hmr: {acc_hmr} cfl: {acc_cfl} dummy: {acc_dummy}")


def ourMethod_init_acc(config):
    ans = {}
    # 加载模型
    model = load_model(config["combination_model_path"])
    # 加载数据
    df_all = pd.read_csv(config["merged_df_path"])
    df_overlap = pd.read_csv(config["merged_overlap_df"])
    df_unique = pd.read_csv(config["merged_unique_df"])
    acc_all = eval_combination_Model(config,df_all)
    acc_overlap = eval_combination_Model(config,df_overlap)
    acc_unique = eval_combination_Model(config,df_unique)
    ans["all_initAcc"] = acc_all
    ans["overlap_initAcc"] = acc_overlap
    ans["unique_initAcc"] = acc_unique
    print("ourMethod_init_acc() successfully!")
    return ans

def test():
    pass

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='3'
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

    # ans = ourMethod_init_acc(config)
    # save_dir = f"exp_data/{dataset_name}"
    # file_name = "initAcc.data"
    # file_path = os.path.join(save_dir,file_name)
    # saveData(ans, file_path)

    # test()

    # get_init_eval(config)

  
    # df_v = get_table_causal_variables()
    # save_dir = "exp_data/all"
    # file_name = "causal_variables.csv"
    # file_path = os.path.join(save_dir, file_name)
    # df_v.to_csv(file_path)

    # calc_correlation()

    # train_acc = get_eval_data()
    # save_dir = f"exp_data/{dataset_name}/retrainResult/percent/OurCombin"
    # file_name = "train_overlap_v3.data"
    # saveData(train_acc, os.path.join(save_dir, file_name))

    # get_Train_Test_Size()
    # df = get_table()
    # save_dir = f"exp_table/{dataset_name}"
    # file_name = "vote.xlsx"
    # file_path = os.path.join(save_dir, file_name)
    # df.to_excel(file_path)
    pass