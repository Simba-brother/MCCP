from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os
from tensorflow.keras.losses import categorical_crossentropy, mse
from tensorflow.keras.optimizers import Adam,Adamax
import sys
sys.path.append("./")
from DataSetConfig import car_body_style_config
import joblib
import tensorflow as tf


os.environ['CUDA_VISIBLE_DEVICES']='3'

def deleteIgnoreFile(file_list):
    '''
    移除隐文件
    '''
    for item in file_list:
        if item.startswith('.'):# os.path.isfile(os.path.join(Dogs_dir, item)):
            file_list.remove(item)
    return file_list

def eval_model(model, classes, df):
    correct_num = 0
    total = df.shape[0]
    test_gen = ImageDataGenerator(rescale=1./255)   # 归一化 importent
    target_size = (224,224) # importent
    batch_size = 32
    # 样本batch
    test_batches = test_gen.flow_from_dataframe(
                                                df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size, class_mode='categorical',
                                                color_mode='rgb', classes = classes, 
                                                shuffle=False, batch_size=batch_size,
                                                validate_filenames=False
                                                )
    print("评估集样本数: {}".format(df.shape[0]))
    for i in range(len(test_batches)):
        batch = next(test_batches)
        X = batch[0]
        Y = batch[1]
        out = model(X, training = False)
        p_label = np.argmax(out, axis = 1)
        ground_label = np.argmax(Y, axis = 1)
        correct_num += np.sum(p_label==ground_label)
    acc = round(correct_num/total, 4)
    # 开始评估
    # loss, acc = model.evaluate_generator(generator=test_batches, steps=test_batches.n/batch_size, verbose = 1)
    return acc

def retrain(model, train_df, epochs):
    train_gen = ImageDataGenerator(rescale=1./255)
    target_size = (224,224)
    batch_size = 5
    # 样本batch
    train_batches = train_gen.flow_from_dataframe(
                                                train_df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size, class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size,
                                                validate_filenames=False
                                                )
    print("训练集样本数: {}".format(train_df.shape[0]))
    model.fit(train_batches,
            steps_per_epoch=train_df.shape[0]//batch_size,
            epochs = epochs,
            verbose = 1,
            shuffle=False
    )
    return model

def kd_retrain(stu, t1, t2, train_df, epoches):
    optimizer=Adamax(learning_rate=1e-3)
    t1_train_gen = config["generator_A"]
    t2_train_gen = config["generator_B"]
    stu_train_gen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   rotation_range=5,
                                   zoom_range=0.15,)
    stu_train_test = ImageDataGenerator(rescale=1./255)

    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    target_size_stu = (224,224)
    batch_size = 5
    t1_bathes = t1_train_gen.flow_from_dataframe(
                            train_df, 
                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                            x_col='file_path', y_col='label', 
                            target_size=target_size_A, class_mode='categorical',
                            color_mode='rgb', classes = None, shuffle=False, 
                            batch_size=batch_size,
                            validate_filenames=False
    )
    t2_bathes = t2_train_gen.flow_from_dataframe(
                        train_df, 
                        directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                        x_col='file_path', y_col='label', 
                        target_size=target_size_B, class_mode='categorical',
                        color_mode='rgb', classes = None, shuffle=False, 
                        batch_size=batch_size,
                        validate_filenames=False
    )
    stu_batches = stu_train_gen.flow_from_dataframe(
                        train_df, 
                        directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                        x_col='file_path', y_col='label', 
                        target_size=target_size_stu, class_mode='categorical',
                        color_mode='rgb', classes = classes, shuffle=False, 
                        batch_size=batch_size,
                        validate_filenames=False)
    stu_batches_test = stu_train_test.flow_from_dataframe(
                        train_df, 
                        directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                        x_col='file_path', y_col='label', 
                        target_size=target_size_stu, class_mode='categorical',
                        color_mode='rgb', classes = classes, shuffle=False, 
                        batch_size=batch_size,
                        validate_filenames=False)
    stu.compile(loss=categorical_crossentropy,optimizer=optimizer,metrics=['accuracy'])
    stu.fit(stu_batches,
            steps_per_epoch=train_df.shape[0]//batch_size,
            epochs = epoches,
            # callbacks=[checkpointer, learning_rate_reduction], #[lr_scheduler, early_stopping], 
            validation_data=stu_batches_test,
            validation_steps = train_df.shape[0]//batch_size,
            verbose = 1,
            shuffle=False)

    # for epoch in range(epoches):
    #     epoch_loss = 0
    #     for i in range(len(stu_batches)):
    #         stu_batch = next(stu_batches)
    #         t1_batch = next(t1_bathes)
    #         t2_batch = next(t2_bathes)
    #         labels = stu_batch[1]
    #         # stu.layers[-1].activation = None 
    #         # t1.layers[-1].activation = None
    #         # t2.layers[-1].activation = None 

    #         t1_out = t1(t1_batch[0], training=False)
    #         t2_out = t2(t2_batch[0], training=False)
    #         out_t1_global = local_to_global(local_to_global_party_A, t1_out)
    #         out_t2_global = local_to_global(local_to_global_party_B, t2_out)
    #         out_t1t2 = np.max(np.array([out_t1_global,out_t2_global]), axis=0)
    #         with tf.GradientTape() as tape:
    #             out_stu = stu(stu_batch[0], training = True)
    #             loss = categorical_crossentropy(out_t1t2, out_stu)
    #             # loss = categorical_crossentropy(labels, out_stu)
    #             epoch_loss += tf.reduce_sum(loss)
    #         gradients = tape.gradient(loss, stu.trainable_weights)
    #         optimizer.apply_gradients(zip(gradients, stu.trainable_weights))
    #     acc = eval_model(stu, classes, merged_test_df)
    #     print(f"训练轮次:{epoch}, 训练轮次损失:{epoch_loss.numpy()}, 训练批次损失:{round(epoch_loss.numpy()/len(stu_batches),4)}, 精度:{acc}")
    return stu

def local_to_global(localToGlobal_dic,proba_local):
    '''
    i方的proba => globa proba
    '''
    predict_value = np.zeros((proba_local.shape[0], all_class_nums))
    mapping = []
    for key, value in localToGlobal_dic.items():
        mapping.append(value)
    predict_value[:, mapping] = proba_local
    return predict_value

def froze_model(model):
    for layer in model.layers[:-1]:
        layer.trainable = False
    return model

def start():
    repeat_num = 1
    common_dir = config["sampled_common_path"]
    sample_num_list = deleteIgnoreFile(os.listdir(common_dir))
    sample_num_list = sorted(sample_num_list, key=lambda e: int(e))
    sample_num_list = [int(sample_num) for sample_num in sample_num_list]
    ans = {}
    for sample_num in sample_num_list:
        if sample_num != 100:
            continue
        ans[sample_num] = []
        cur_dir = os.path.join(common_dir, str(sample_num))
        for repeat in range(repeat_num):
            csv_file_name = "sampled_"+str(repeat)+".csv"
            csv_file_path = os.path.join(cur_dir, csv_file_name)
            retrain_df = pd.read_csv(csv_file_path)
            # 重新加载模型
            my_model = load_model("/data/mml/some_models/Xception_224_224_claNum_10.h5")
            # 冻结预训练层
            my_model = froze_model(my_model)
            # my_model = retrain(my_model, retrain_df, epochs=5)
            my_model = kd_retrain(my_model, model_A, model_B, merged_test_df, epoches=30)
            acc = eval_model(my_model, classes, merged_test_df)
            acc_improve = round(acc - init_acc,4)
            print("目标采样百分比:{}%, 实际采样数量:{}, 重复次数:{}, 混合评估精度提高:{}".format(sample_num, 
                                                                                    retrain_df.shape[0],
                                                                                    repeat, 
                                                                                    acc_improve, 
                                                                                    ))
            ans[sample_num].append({"acc_improve":acc_improve})
    return ans

def get_myModel(pretrained_model, class_num):
    # model_cut = Model(inputs = pretrained_model.input, outputs = pretrained_model.get_layer(index = -2).output)
    new_output = Dense(class_num, activation='softmax', name="new_Dense")(pretrained_model.output)
    my_model = Model(inputs = pretrained_model.input, outputs = new_output)
    return my_model

# 全局变量
config = car_body_style_config
# 双方的local to global
local_to_global_party_A = joblib.load(config["local_to_global_party_A_path"])
local_to_global_party_B = joblib.load(config["local_to_global_party_B_path"])
# 预训练model
pretrained_model = load_model("/data/mml/some_models/Xception_224_224.h5")
# teacher model
model_A = load_model("/data/mml/overlap_v2_datasets/car_body_style/party_A/models/model_struct/Xception.h5")
model_A.load_weights("/data/mml/overlap_v2_datasets/car_body_style/party_A/models/model_weights/model_weight_041_0.8352.h5")
model_B = load_model("/data/mml/overlap_v2_datasets/car_body_style/party_B/models/model_struct/Xception.h5")
model_B.load_weights("/data/mml/overlap_v2_datasets/car_body_style/party_B/models/model_weights/model_weight_039_0.8464.h5")
# 融合的测试集
merged_test_df = pd.read_csv("/data/mml/overlap_v2_datasets/car_body_style/merged_data/test/merged_df.csv")
merged_train_df = pd.read_csv("/data/mml/overlap_v2_datasets/car_body_style/merged_data/train/merged_df.csv")
# model的classes
classes = list(merged_train_df["label"].unique())
classes.sort()
all_class_nums = len(classes)
# 加载model
# my_model = get_myModel(pretrained_model, class_num=10)
#  my_model.save("/data/mml/some_models/Xception_224_224_claNum_10.h5")
my_model = load_model("/data/mml/overlap_v2_datasets/car_body_style/merged_model/combination_model_inheritWeights.h5")
# 评估模型
# 编译model
# my_model.compile(loss=categorical_crossentropy,optimizer=Adam(learning_rate=1e-4),metrics=['accuracy'])
init_acc = eval_model(my_model, classes, merged_test_df)
print(f"init_acc:{init_acc}")

if __name__ == "__main__":
    ans = start()
    print(ans)