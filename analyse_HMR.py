from tensorflow.keras.models import Model, Sequential, load_model
import pandas as pd
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, Adamax
from DataSetConfig import car_body_style_config
import numpy as np
from utils import deleteIgnoreFile, saveData
from sklearn.model_selection import train_test_split
import joblib



def getClasses(dir_path):
    '''
    得到分类目录的分类列表
    '''
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    classes_name_list.sort()  # 字典序
    return classes_name_list

def eval_model(model, df_test, test_gen, classes, target_size):
    total = df_test.shape[0]
    correct_num = 0
    batch_size = 5
    test_batches = test_gen.flow_from_dataframe(
                                                df_test, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size, class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size,
                                                validate_filenames=False
                                                )

    print("评估集样本数: {}".format(total))
    for i in range(len(test_batches)):
        batch = next(test_batches)
        X = batch[0]
        Y = batch[1] # one hot
        # model.layers[-1].activation = None
        out = model(X, training=False)
        p_label = np.argmax(out, axis = 1)
        ground_label = np.argmax(Y, axis = 1)
        correct_num += np.sum(p_label==ground_label)
    acc = round(correct_num/total, 4)
    # 开始评估
    # loss, acc = model.evaluate_generator(generator=test_batches,steps=test_batches.n / batch_size, verbose = 1)
    return acc

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
    model_cut = Model(inputs = model.input, outputs = model.get_layer(index = -2).output)
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

def retrain_model(model, df_train, train_gen, classes, target_size):
    batch_size = 5
    epochs = 5
    train_batches = train_gen.flow_from_dataframe(
                                                df_train, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size, 
                                                class_mode='categorical',
                                                color_mode='rgb', classes = classes, 
                                                shuffle=True, batch_size=batch_size,
                                                validate_filenames=False
                                                )
    history = model.fit(train_batches,
                        steps_per_epoch=df_train.shape[0]//batch_size,
                        epochs = epochs,
                        # callbacks=[checkpointer, learning_rate_reduction], #[lr_scheduler, early_stopping], 
                        # validation_data=batches_test,
                        # validation_steps = merged_df.shape[0]//batch_size,
                        verbose = 1,
                        shuffle=True)
    return model

def eval_extended_model(extended_model, test_gen, df_test, target_size, classes):
    total = df_test.shape[0]
    correct_num = 0
    batch_size = 5
    test_batches = test_gen.flow_from_dataframe(
                                                df_test, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size, class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size,
                                                validate_filenames=False
                                                )
    # loss, acc = extended_model.evaluate_generator(generator=test_batches,steps=test_batches.n / batch_size, verbose = 1)
    for i in range(len(test_batches)):
        batch = next(test_batches)
        X = batch[0]
        Y = batch[1] # one hot
        # extended_model.layers[-1].activation = None
        out = extended_model(X, training=False)
        # out = extended_model.predict(X, batch_size=None, verbose=0)
        out_cut = out[:,:-1]
        p_label = np.argmax(out_cut, axis = 1)
        ground_label = np.argmax(Y, axis = 1)
        correct_num += np.sum(p_label==ground_label)
    acc = round(correct_num/total, 4)
    return acc

def get_retrain_df(df, flag):
    if flag == "A":
        return df[df["source"]==1]
    if flag == "B":
        return df[df["source"]==2]
    if flag == "A_overlap_unique":
        df_A = df[df["source"]==1]
        df_A.loc[df_A["is_overlap"]==0, "label"] = "other"
        return df_A
    if flag == "B_overlap_unique":
        df_B = df[df["source"]==2]
        df_B.loc[df_B["is_overlap"]==0, "label"] = "other"
        return df_B
    if flag == "all_analyse_A":
        df.loc[(df["source"]!=1)&(df["is_overlap"]==0), "label"] = "other"    
        return df
    if flag == "all_analyse_B":
        df.loc[(df["source"]!=2)&(df["is_overlap"]==0), "label"] = "other"    
        return df
    if flag == "A_B_overlap":
        df_A_B_overlap = df[(df["source"]==1)|((df["source"]==2)&(df["is_overlap"]==1))]
        return df_A_B_overlap
    if flag == "B_A_overlap":
        df_B_A_overlap = df[(df["source"]==2)|((df["source"]==1)&(df["is_overlap"]==1))]
        return df_B_A_overlap
    if flag == "A_B_unique":
        df_A_B_unique = df[(df["source"]==1)|((df["source"]==2)&(df["is_overlap"]==0))]
        df_A_B_unique.loc[(df_A_B_unique["source"]!=1)&(df_A_B_unique["is_overlap"]==0), "label"] = "other"
        return df_A_B_unique
    if flag == "B_A_unique":
        df_B_A_unique = df[(df["source"]==2)|((df["source"]==1)&(df["is_overlap"]==0))]
        df_B_A_unique.loc[(df_B_A_unique["source"]!=2)&(df_B_A_unique["is_overlap"]==0), "label"] = "other"
        return df_B_A_unique
    if flag == "B_overlap":
        df_B_overlap = df[((df["source"]==2)&(df["is_overlap"]==1))]
        return df_B_overlap
    if flag == "B_unique":
        df_B_unique = df[((df["source"]==2)&(df["is_overlap"]==0))]
        df_B_unique["label"] = "other"
        return df_B_unique
    if flag == "A_overlap":
        df_A_overlap = df[((df["source"]==1)&(df["is_overlap"]==1))]
        return df_A_overlap
    if flag == "A_unique":
        df_A_unique = df[((df["source"]==1)&(df["is_overlap"]==0))]
        df_A_unique["label"] = "other"
        return df_A_unique
    
def analyse_A():
    ans = {}
    # 加载model_A model
    model_A = load_model(config['model_A_struct_path'])
    model_A.load_weights(config['model_A_weight_path'])

    # 评估model_A test_A performance
    df_test_A = pd.read_csv(config["df_eval_party_A_path"])
    test_gen = config["generator_A_test"]
    classes = getClasses(config["dataset_A_train_path"])
    target_size = config["target_size_A"]
    acc = eval_model(model_A, df_test_A, test_gen, classes, target_size)
    print(f"model_A 原始acc:{round(acc,4)}")

    # extend model_B
    extended_model_A = add_reserve_class(model_A)
    extended_model_A.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=1e-5),metrics=['accuracy'])
    # eval extended_model_B test_B performance
    df_test_A = pd.read_csv(config["df_eval_party_A_path"])
    test_gen = config["generator_A_test"]
    classes = getClasses(config["dataset_A_train_path"])
    classes.append("zzz")
    target_size = config["target_size_A"]
    base_acc = eval_extended_model(extended_model_A, test_gen, df_test_A,  target_size, classes)
    print(f"extended model A训练前acc:{base_acc}")

    common_dir = config["sampled_common_path"]
    sample_num_list = deleteIgnoreFile(os.listdir(common_dir))
    sample_num_list = sorted(sample_num_list, key=lambda e: int(e))
    sample_num_list = [int(sample_num) for sample_num in sample_num_list]
    repeat_num = 5  # 先 统计 5 次随机采样 importent

    for sample_num in sample_num_list:
        ans[sample_num] = []
        cur_dir = os.path.join(common_dir, str(sample_num))
        for repeat in range(repeat_num):
            csv_file_name = "sampled_"+str(repeat)+".csv"
            csv_file_path = os.path.join(cur_dir, csv_file_name)
            # 加载标记代价采样集        
            retrain_df = pd.read_csv(csv_file_path)
            print(f"采样比例:{sample_num}%, 实际采样数量:{retrain_df.shape[0]}, 重复实验idx:{repeat}")
            # retrain extended_model_A
            retrain_df = get_retrain_df(retrain_df, "all_analyse_A")
            train_gen = config["generator_A"]
            classes = getClasses(config["dataset_A_train_path"])
            classes.append("other")
            model_A = load_model(config['model_A_struct_path'])
            model_A.load_weights(config['model_A_weight_path'])
            extended_model_A = add_reserve_class(model_A)
            extended_model_A.compile(Adamax(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy']) 
            extended_model_A = retrain_model(extended_model_A, retrain_df, train_gen, classes, target_size)

            # eval extended_model_A on local_A
            df_test_A = pd.read_csv(config["df_eval_party_A_path"])
            test_gen = config["generator_A_test"]
            target_size = config["target_size_A"]
            classes = getClasses(config["dataset_A_train_path"])
            classes.append("other")
            acc = eval_extended_model(extended_model_A, test_gen, df_test_A, target_size, classes)
            improve_acc = round(acc-base_acc,4)
            print(f"retrain后extended_model_A在local_A上acc:{acc}, improve_acc:{improve_acc}")
            ans[sample_num].append(improve_acc)
    print(ans)
    save_dir = "exp_data/car_body_style/HMR/analyse_model_A_calibrate"
    save_file_name = "analyse_A.data"
    save_path = os.path.join(save_dir, save_file_name)
    saveData(ans, save_path)
    print(f"saved success! save_path:{save_path}")
 
def analyse_B():
    ans = {}
    # 加载model_B model
    model_B = load_model(config['model_B_struct_path'])
    model_B.load_weights(config['model_B_weight_path'])

    # 评估model_B test_B performance
    df_test_B = pd.read_csv(config["df_eval_party_B_path"])
    test_gen = config["generator_B_test"]
    classes = getClasses(config["dataset_B_train_path"])
    target_size = config["target_size_B"]
    acc = eval_model(model_B, df_test_B, test_gen, classes, target_size)
    print(f"model_B 原始acc:{acc}")

    # extend model_B
    extended_model_B = add_reserve_class(model_B)
    extended_model_B.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=1e-5),metrics=['accuracy'])
    # eval extended_model_B test_B performance
    df_test_B = pd.read_csv(config["df_eval_party_B_path"])
    test_gen = config["generator_B_test"]
    classes = getClasses(config["dataset_B_train_path"])
    classes.append("zzz")
    target_size = config["target_size_B"]
    base_acc = eval_extended_model(extended_model_B, test_gen, df_test_B,  target_size, classes)
    print(f"extended model B训练前acc:{base_acc}")

    common_dir = config["sampled_common_path"]
    sample_num_list = deleteIgnoreFile(os.listdir(common_dir))
    sample_num_list = sorted(sample_num_list, key=lambda e: int(e))
    sample_num_list = [int(sample_num) for sample_num in sample_num_list]
    repeat_num = 5  # 先 统计 5 次随机采样 importent

    for sample_num in sample_num_list:
        ans[sample_num] = []
        cur_dir = os.path.join(common_dir, str(sample_num))
        for repeat in range(repeat_num):
            csv_file_name = "sampled_"+str(repeat)+".csv"
            csv_file_path = os.path.join(cur_dir, csv_file_name)
            # 加载标记代价采样集        
            retrain_df = pd.read_csv(csv_file_path)
            print(f"采样比例:{sample_num}%, 实际采样数量:{retrain_df.shape[0]}, 重复实验idx:{repeat}")
            # retrain extended_model_B
            retrain_df = get_retrain_df(retrain_df, "all_analyse_B")
            train_gen = config["generator_B"]
            classes = getClasses(config["dataset_B_train_path"])
            classes.append("other")
            model_B = load_model(config['model_B_struct_path'])
            model_B.load_weights(config['model_B_weight_path'])
            extended_model_B = add_reserve_class(model_B)
            extended_model_B.compile(Adamax(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy']) 
            extended_model_B = retrain_model(extended_model_B, retrain_df, train_gen, classes, target_size)

            # eval extended_model_B on local_B
            df_test_B = pd.read_csv(config["df_eval_party_B_path"])
            test_gen = config["generator_B_test"]
            target_size = config["target_size_B"]
            classes = getClasses(config["dataset_B_train_path"])
            classes.append("other")
            acc = eval_extended_model(extended_model_B, test_gen, df_test_B, target_size, classes)
            improve_acc = round(acc-base_acc,4)
            print(f"retrain后extended_model_B在local_B上acc:{acc}, improve_acc:{improve_acc}")
            ans[sample_num].append(improve_acc)
    print(ans)
    save_dir = "exp_data/car_body_style/HMR/analyse_model_B_calibrate"
    save_file_name = "analyse_B_new.data"
    save_path = os.path.join(save_dir, save_file_name)
    saveData(ans, save_path)
    print(f"saved success! save_path:{save_path}")

# def eval_compare():
#     # 加载model_A model
#     model_A = load_model(config['model_A_struct_path'])
#     model_A.load_weights(config['model_A_weight_path'])
#     extended_model = add_reserve_class(model_A)
#     # 评估model_A test_A performance
#     df_test_A = pd.read_csv(config["df_eval_party_A_path"])
#     test_gen = config["generator_A_test"]
#     classes = getClasses(config["dataset_A_train_path"])
#     classes_extend = []
#     classes_extend.extend(classes)
#     classes_extend.append("qther")
#     target_size = config["target_size_A"]

#     total = df_test_A.shape[0]
#     correct_num = 0
#     batch_size = 5
#     test_batches_1 = test_gen.flow_from_dataframe(
#                                                 df_test_A, 
#                                                 directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
#                                                 x_col='file_path', y_col='label', 
#                                                 target_size=target_size, class_mode='categorical',
#                                                 color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size,
#                                                 validate_filenames=False
#                                                 )
#     test_batches_2 = test_gen.flow_from_dataframe(
#                                             df_test_A, 
#                                             directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
#                                             x_col='file_path', y_col='label', 
#                                             target_size=target_size, class_mode='categorical',
#                                             color_mode='rgb', classes = classes_extend, shuffle=False, batch_size=batch_size,
#                                             validate_filenames=False
#                                             )
#     # loss, acc = extended_model.evaluate_generator(generator=test_batches,steps=test_batches.n / batch_size, verbose = 1)
#     for i in range(len(test_batches_1)):
#         batch_1 = next(test_batches_1)
#         batch_2 = next(test_batches_2)
#         X = batch_1[0]
#         Y_1 = batch_1[1] # one hot
#         Y_2 = batch_2[1] # one hot
#         model_A.layers[-1].activation = None
#         extended_model.layers[-1].activation = None
#         out_1 = model_A(X, training=False)
#         out_2 = extended_model(X, training=False)
#         out_2_cut = out_2[:,:-1]
#         p_label_1 = np.argmax(out_1, axis = 1)
#         p_label_2 = np.argmax(out_2_cut, axis = 1)
#         ground_label_1 = np.argmax(Y_1, axis = 1)
#         ground_label_2 = np.argmax(Y_2, axis = 1)
#         for j in range(p_label_1.shape[0]):
#             if p_label_1[j] != p_label_2[j]:
#                 print('看看')
#         for k in range(ground_label_1.shape[0]):
#             if ground_label_1[k] != ground_label_2[k]:
#                 print("再看看")
#         # correct_num += np.sum(p_label==ground_label_1)
#     # acc = round(correct_num/total, 4)
#     return 0

# 全局变量
# 设置训练显卡
os.environ['CUDA_VISIBLE_DEVICES']='4'
config = car_body_style_config

if __name__ == "__main__":
    # analyse_A()   
    # analyse_B()
    # eval_compare()
    pass
