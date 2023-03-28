import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import optimizers
import sys
sys.path.append("/home/mml/workspace/model_reuse_v2/")
from utils import deleteIgnoreFile, saveData
import joblib
# 加载数据集 config
from DataSetConfig import food_config, fruit_config, sport_config, weather_config, flower_2_config, car_body_style_config, animal_config, animal_2_config, animal_3_config
# 设置训练显卡
os.environ['CUDA_VISIBLE_DEVICES']='3'
# 配置变量
config = sport_config

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

def start_reTrain():
    common_dir = config["sampled_common_path"]
    repeat_num = 5  # 先 统计 5 次随机采样 importent
    merged_csv_path = config["merged_df_path"]
    combination_model_path = config["combination_model_path"]
    # A,B双发的数据生成器 训练集
    # flower
    # generator_left = ImageDataGenerator(rescale = 1./255) 
    # generator_right = ImageDataGenerator(rescale = 1./255 , 
    #                             rotation_range=30 ,
    #                             zoom_range=0.2,
    #                             horizontal_flip=True,
    #                             brightness_range=[0.6,1],
    #                             fill_mode='nearest',)

    # food
    # generator_left = ImageDataGenerator(horizontal_flip=True,rotation_range=20, width_shift_range=.2,
    #                                 height_shift_range=.2, zoom_range=.2)
    # generator_right = ImageDataGenerator(rescale = 1./255)

    # Fruit
    # generator_left = ImageDataGenerator(rescale = 1./255, rotation_range=20, width_shift_range=.2,
    #                                 height_shift_range=.2, zoom_range=.2)
    # generator_right = ImageDataGenerator(rescale = 1./255)

    # sport
    # generator_left = ImageDataGenerator(horizontal_flip=True,rotation_range=20, width_shift_range=.2,
    #                                 height_shift_range=.2, zoom_range=.2)
    # generator_right = ImageDataGenerator(rescale=1./255, zoom_range=0.5,horizontal_flip=True,rotation_range=40,vertical_flip=0.5,width_shift_range=0.3,height_shift_range=0.2,brightness_range=[0.2,1.0],fill_mode='nearest')

    # car_body_style
    # generator_left =ImageDataGenerator(rescale=1./255,
    #                             horizontal_flip=True,
    #                             width_shift_range=0.15,
    #                             height_shift_range=0.15,
    #                             rotation_range=5,
    #                             zoom_range=0.15)  # 归一化
    # generator_right =ImageDataGenerator(rescale=1./255,
    #                     horizontal_flip=True,
    #                     width_shift_range=0.15,
    #                     height_shift_range=0.15,
    #                     rotation_range=5,
    #                     zoom_range=0.15)  # 归一化

    # animal
    # generator_left =ImageDataGenerator(rescale=1./255,
    #                             horizontal_flip=True,
    #                             width_shift_range=0.15,
    #                             height_shift_range=0.15,
    #                             rotation_range=5,
    #                             zoom_range=0.15)  # 归一化
    # generator_right =ImageDataGenerator(rescale=1./255)  # 归一化

    # weather
    # generator_left =ImageDataGenerator(
    #                             horizontal_flip=True,
    #                             vertical_flip=True,
    #                             rotation_range=10,
    #                             width_shift_range=0.2,
    #                             height_shift_range=0.2,
    #                             zoom_range=.2)  # 归一化
    # generator_right =ImageDataGenerator(
    #                             rescale=1./255,
    #                             horizontal_flip=True,
    #                             rotation_range=10)  # 归一化

    generator_left = config["generator_A"]
    generator_right = config["generator_B"]
    generator_left_test = config["generator_A_test"]
    generator_right_test = config["generator_B_test"]
    # A,B双方的数据生成器-测试集

    # flower
    # generator_left_test = ImageDataGenerator(rescale = 1./255)  # 归一化
    # generator_right_test = ImageDataGenerator(rescale = 1./255)  # 归一化

    # food
    # generator_left_test =ImageDataGenerator() 
    # generator_right_test = ImageDataGenerator(rescale = 1./255)

    # Fruit
    # generator_left_test = ImageDataGenerator(rescale = 1./255)
    # generator_right_test = ImageDataGenerator(rescale = 1./255)

    # sport
    # generator_left_test = ImageDataGenerator()
    # generator_right_test = ImageDataGenerator(rescale=1./255)

    # car_body_style
    # generator_left_test =ImageDataGenerator(rescale=1./255) 
    # generator_right_test =ImageDataGenerator(rescale=1./255) 

    # animal
    # generator_left_test =ImageDataGenerator(rescale=1./255) 
    # generator_right_test =ImageDataGenerator(rescale=1./255) 

    # weather
    # generator_left_test =ImageDataGenerator() 
    # generator_right_test =ImageDataGenerator(rescale=1./255)

    # 全局类别,字典序列
    merged_df = pd.read_csv(merged_csv_path)

    classes = merged_df["label"].unique()
    classes = np.sort(classes).tolist()
    batch_size = 8
    target_size_A = config["target_size_A"] # flower (224,224), food:(256,256), Fruit:(224,224), sport:(224,224), car_body_style:(256,256), animal:(224,224), weather:(100,100)
    target_size_B = config["target_size_B"] # flower (150,150), food:(224,224), Fruit:(224,224), sport:(224,224), car_body_style:(224,224), animal:(150,150), weather:(256,256)
    # 模型初始精度,重训练好后减去它
    # importent
    combination_model = load_model(combination_model_path)
    combination_model.compile(optimizer="adam",loss="categorical_crossentropy",metrics="accuracy")
    init_val_acc = eval_combination_model(combination_model, merged_df, generator_left, generator_right, classes, target_size_A, target_size_B)
    init_val_acc = round(init_val_acc,4)  # flower: 0.7477, food:0.8170, Fruit:0.7051, sport: 0.7517, car_body_style:0.8013, animal:0.8150, weather: 0.7568
    # learning_rate:{flower:0.0003, food:0.00003, Fruit:.0003, sport: 0.00003, car_body_style:0.0003, animal: 0.0003, weather: 0.0003, animal_2:0.0005}
    # importent
    combination_learning_rate = config["combiantion_lr"]
    # 训练轮次
    epochs=5
    sample_num_list = deleteIgnoreFile(os.listdir(common_dir))
    sample_num_list = sorted(sample_num_list, key=lambda e: int(e))
    improve_Acc_result = {}
    improve_Acc_records_result = {}
    for sample_num in sample_num_list:
        cur_dir = os.path.join(common_dir, sample_num)
        # 用于记录 该 抽样 数量 下 重复 次数 val_acc。 用于求平均
        record_list = []
        for repeat in range(repeat_num):
            csv_file_name = "sampled_"+str(repeat)+".csv"
            csv_file_path = os.path.join(cur_dir, csv_file_name)
            df = pd.read_csv(csv_file_path)
            print("采样数量: {}, 实际采样数量: {}, repeat: {}".format(sample_num, df.shape[0],repeat))
            # A,B双方的batches 训练集
            y_col = "label"  # importent!!!!!!
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

            batches_train = generate_generator_multiple(batches_A, batches_B)


            # A,B双方的batches 测试集
            batches_A_test = generator_left_test.flow_from_dataframe(merged_df, 
                                                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                            # subset="training",
                                                            seed=42,
                                                            x_col='file_path', y_col='label', 
                                                            target_size=target_size_A, class_mode='categorical',
                                                            color_mode='rgb', classes=classes, shuffle=False, batch_size=batch_size,
                                                            validate_filenames=False)
            batches_B_test = generator_right_test.flow_from_dataframe(merged_df, 
                                                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                            # subset="training",
                                                            seed=42,
                                                            x_col='file_path', y_col='label', 
                                                            target_size=target_size_B, class_mode='categorical',
                                                            color_mode='rgb', classes=classes, shuffle=False, batch_size=batch_size,
                                                            validate_filenames=False)

            batches_test = generate_generator_multiple(batches_A_test, batches_B_test)
            # 加载模型
            model = load_model(combination_model_path)
            # 冻结之前的层
            # for layer in model.layers[:-2]:
            #     layer.trainable = False
            # 模型编译 
            adam = optimizers.Adam(learning_rate=combination_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")
            # 开始训练
            history = model.fit(batches_train,
                            steps_per_epoch=df.shape[0]//batch_size,
                            epochs = epochs,
                            # callbacks=[checkpointer, learning_rate_reduction], #[lr_scheduler, early_stopping], 
                            validation_data=batches_test,
                            validation_steps = merged_df.shape[0]//batch_size,
                            verbose = 1,
                            shuffle=False)
            dic_metric = history.history
            val_accuracy = dic_metric["val_accuracy"][epochs-1]
            record_list.append(round(val_accuracy, 4))
        
        avg_val_acc = np.mean(record_list)
        improve_acc = avg_val_acc - init_val_acc
        improve_acc = round(improve_acc,4)
        print("采样数量: {}, 平均精度提升值: {}".format(sample_num, improve_acc))
        improve_Acc_result[int(sample_num)] = improve_acc
        improve_Acc_records_result[int(sample_num)] = np.array(record_list)-init_val_acc
    save_dir = config["save_retrainResult_path"]
    save_file_name = "reTrain_acc_improve.data"
    saveData(improve_Acc_result, os.path.join(save_dir, save_file_name))
    saveData(improve_Acc_records_result, os.path.join(save_dir, "reTrain_acc_improve_accords.data"))
    print("return:", improve_Acc_result)
    print("start_reTrain() success")
    
'''
创建该文件时的demo方法
'''
def demo():
    # 加载数据集(重训练用到的采样集)
    csv_path = "exp_data/flower/sampling/random/30/sampled_0.csv"
    df = pd.read_csv(csv_path)
    # 加载重训练用到的评估集(其实就是混合的双方测试集)
    merged_csv_path = "/data/mml/overlap_v2_datasets/flower/merged_data/test/merged_df.csv"
    merged_df = pd.read_csv(merged_csv_path) 

    # A,B双发的数据生成器 训练集
    generator_left = ImageDataGenerator(rescale = 1./255)  # 归一化
    generator_right = ImageDataGenerator(rescale = 1./255 , 
                                        rotation_range=30 ,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        brightness_range=[0.6,1],
                                        fill_mode='nearest',
                                    )  # 归一化
    # A,B双发的数据生成器 测试集
    generator_left_test = ImageDataGenerator(rescale = 1./255)  # 归一化
    generator_right_test = ImageDataGenerator(rescale = 1./255)  # 归一化

    # 全局类别,字典序列
    classes = merged_df["label"].unique()
    classes = np.sort(classes).tolist()
    batch_size = 8
    target_size_A = (224,224)
    target_size_B = (150,150)
    # A,B双方的batches 训练集
    batches_A = generator_left.flow_from_dataframe(df, 
                                                    directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                    # subset="training",
                                                    seed=42,
                                                    x_col='file_path', y_col='label', 
                                                    target_size=target_size_A, class_mode='categorical',
                                                    color_mode='rgb', classes=classes, shuffle=False, batch_size=batch_size)
    batches_B = generator_right.flow_from_dataframe(df, 
                                                    directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                    # subset="training",
                                                    seed=42,
                                                    x_col='file_path', y_col='label', 
                                                    target_size=target_size_B, class_mode='categorical',
                                                    color_mode='rgb', classes=classes, shuffle=False, batch_size=batch_size)

    batches = generate_generator_multiple(batches_A, batches_B)

    # A,B双方的batches 测试集
    batches_A_test = generator_left_test.flow_from_dataframe(merged_df, 
                                                    directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                    # subset="training",
                                                    seed=42,
                                                    x_col='file_path', y_col='label', 
                                                    target_size=target_size_A, class_mode='categorical',
                                                    color_mode='rgb', classes=classes, shuffle=False, batch_size=batch_size)
    batches_B_test = generator_right_test.flow_from_dataframe(merged_df, 
                                                    directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                    # subset="training",
                                                    seed=42,
                                                    x_col='file_path', y_col='label', 
                                                    target_size=target_size_B, class_mode='categorical',
                                                    color_mode='rgb', classes=classes, shuffle=False, batch_size=batch_size)

    batches_test = generate_generator_multiple(batches_A_test, batches_B_test)


    # 加载模型
    model_path = "/data/mml/overlap_v2_datasets/flower/merged_model/combination_model_inheritWeights.h5"
    model = load_model(model_path)
    # 模型编译
    adam = optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")
    # 开始训练
    epochs = 5
    history = model.fit(batches,
                    steps_per_epoch=df.shape[0]//batch_size,
                    epochs=epochs, 
                    # callbacks=[checkpointer, learning_rate_reduction], #[lr_scheduler, early_stopping], 
                    validation_data=batches_test,
                    validation_steps = merged_df.shape[0]//batch_size,
                    verbose = 1,
                    shuffle=False)

    dic_metric = history.history
    val_accuracy = dic_metric["val_accuracy"][epochs-1]
    print("demo success")

if __name__ == "__main__":
    # start_reTrain()
    a = joblib.load("exp_data/food/retrainResult/percent/random/reTrain_acc_improve.data")
    print(a)
    pass