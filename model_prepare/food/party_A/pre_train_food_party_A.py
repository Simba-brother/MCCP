'''
准备预训练模型 
ref_code: https://www.kaggle.com/code/jiaowoguanren/food-11-image-classification-dataset-tensorflow
          https://www.kaggle.com/code/gpiosenka/food-f1-score-90 (训练代码比较详细)
'''
import imp
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# keras.models
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import Callback
# keras.layers
from tensorflow.keras.layers import Layer, Dense, Conv2D, MaxPool2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Dropout
# keras.callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import Model
from utils import deleteIgnoreFile
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax
import tensorflow as tf
import time
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def getClasses(dir_path):
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    classes_name_list.sort()
    return classes_name_list




def start_train():
    # 加载数据
    train_csv_path = "/data/mml/overlap_v2_datasets/food/party_A/dataset_split/train.csv"
    val_csv_path = "/data/mml/overlap_v2_datasets/food/party_A/dataset_split/val.csv"
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # 声明出一个生成器
    train_gen =ImageDataGenerator(horizontal_flip=True,rotation_range=20, width_shift_range=.2,
                                    height_shift_range=.2, zoom_range=.2, validation_split=0.2)  # 归一化
    val_gen =ImageDataGenerator() 

    # 获得batches
    classes = getClasses("/data/mml/overlap_v2_datasets/food/party_A/dataset_split/train")
    train_df = train_df.sample(frac = 1)
    train_batches = train_gen.flow_from_dataframe(train_df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                subset="training",
                                                x_col='file_path', y_col='label', 
                                                target_size=(256,256), class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=True, batch_size=32)

    val_batches = train_gen.flow_from_dataframe(train_df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                subset= "validation",
                                                x_col='file_path', y_col='label', 
                                                target_size=(256,256), class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=True, batch_size=32)
    test_batches = val_gen.flow_from_dataframe(val_df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=(256,256), class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=False, batch_size=32)

    print(train_batches.n)
    print(val_batches.n)
    print(test_batches.n)

    # 加载base_model
    pre_trained_model = load_model("/data/mml/overlap_v2_datasets/food/party_A/models/standard_base_model/base_model_EfficientNetB3.h5")
    pre_trained_model.trainable=True
    x=pre_trained_model.output
    x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
    x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                    bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
    x=Dropout(rate=.4, seed=123)(x)       
    output=Dense(11, activation='softmax')(x)
    model=Model(inputs=pre_trained_model.input, outputs=output)
    lr=.001 # start with this learning rate
    # 编译模型
    model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy']) 
    # 保存搭建好的模型结构
    model.save("/data/mml/overlap_v2_datasets/food/party_A/models/model_struct/EfficientNetB3.h5")
    # 设置检查点
    saved_dir_path = "/data/mml/overlap_v2_datasets/food/party_A/models/model_weights"
    checkpointer = ModelCheckpoint(os.path.join(saved_dir_path, 'model_weight_{epoch:03d}_{val_accuracy:.4f}.h5'),
                                    verbose=1, 
                                    monitor='val_accuracy',
                                    save_weights_only=True, 
                                    save_best_only=True)
    # 设置学习率减小
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 3, verbose=1,factor=0.6, min_lr=0.000001)
    
    # 开始训练
    history = model.fit(train_batches,
                    # steps_per_epoch=reTrain_set_size/batch_size,
                    epochs=40, 
                    callbacks=[checkpointer], #[lr_scheduler, early_stopping, learning_rate_reduction], 
                    validation_data=val_batches,
                    # validation_steps = reValid_df_set_size/batch_size,
                    verbose = 1,
                    shuffle=True)


if __name__ == "__main__":
    start_train()