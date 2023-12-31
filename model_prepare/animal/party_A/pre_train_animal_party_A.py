'''
准备预训练模型 
refcode: https://www.kaggle.com/code/timothymulenga/african-wildlife-animal-classification-fastai
'''
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# keras.models
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import Callback
# keras.layers
from tensorflow.keras.layers import Layer, Dense, Conv2D, MaxPool2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Dropout
# keras.callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import Model
import sys
sys.path.append("./")
from utils import deleteIgnoreFile
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax
import tensorflow as tf
import time
import numpy as np
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def getClasses(dir_path):
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    return classes_name_list

def start_train():
    # 加载数据
    train_csv_path = "/data/mml/overlap_v2_datasets/animal/party_A/dataset_split/train.csv"
    val_csv_path = "/data/mml/overlap_v2_datasets/animal/party_A/dataset_split/val.csv"
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # 声明出一个生成器
    train_gen =ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   rotation_range=5,
                                   zoom_range=0.15,
                                   validation_split=0.2)  # 归一化
    val_gen =ImageDataGenerator(rescale=1./255) 

    # 获得batches
    classes = getClasses("/data/mml/overlap_v2_datasets/animal/party_A/dataset_split/train")
    train_df = train_df.sample(frac = 1)
    target_size = (224,224)
    train_batches = train_gen.flow_from_dataframe(train_df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                subset="training",
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size, class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=True, batch_size=32)

    val_batches = train_gen.flow_from_dataframe(train_df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                subset= "validation",
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size, class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=True, batch_size=32)
    test_batches = val_gen.flow_from_dataframe(val_df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size, class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=False, batch_size=32)

    print(train_batches.n)
    print(val_batches.n)
    print(test_batches.n)

    # 加载base_model
    base_model = load_model("/data/mml/overlap_v2_datasets/animal/party_A/models/standard_base_model/base_model_VGG19_224_224.h5")
    base_model.trainable = False
    x=base_model.output
    x= BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, name = "party_A_BacthNormal")(x)
    x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                    bias_regularizer=regularizers.l1(0.006) ,activation='relu', name = "party_A_Dense_1")(x)
    x=Dropout(rate=.45, seed=123, name = "party_A_Dropout_1")(x)        
    output=Dense(4, activation='softmax', name = "party_A_Dense_2")(x)
    model=Model(inputs=base_model.input, outputs=output)
    model.compile(Adamax(lr=.001), loss='categorical_crossentropy', metrics=['accuracy']) 

    # pre_trained_model = load_model("/data/mml/overlap_v2_datasets/car_body_style/party_A/models/standard_base_model/base_model_Xception.h5")
    # pre_trained_model.trainable = True
    # model = Sequential([
    #     pre_trained_model,
    #     MaxPool2D((2,2) , strides = 2),
    #     Flatten(),
    #     Dense(512, activation='relu'),
    #     Dropout(rate=0.5, seed=123),
    #     Dense(256, activation='relu'),
    #     Dropout(rate=0.5, seed=123),
    #     Dense(128, activation='relu'),
    #     Dropout(rate=0.5, seed=123),
    #     Dense(7 , activation='softmax')])


    # pre_trained_model.trainable=True
    # x=pre_trained_model.output
    # x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
    # x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
    #                 bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
    # x=Dropout(rate=.4, seed=123)(x)       
    # output=Dense(11, activation='softmax')(x)
    # model=Model(inputs=pre_trained_model.input, outputs=output)
    # class MyBiTModel(Model):
    #     def __init__(self, num_classes, module):
    #         super().__init__()

    #         self.num_classes = num_classes
    #         self.dense1 = Dense(512, activation='relu')
    #         self.dense2 = Dense(256, activation='relu')
    #         self.dense3 = Dense(128, activation='relu')
    #         self.dropout = Dropout(0.5)
    #         self.head = Dense(num_classes, kernel_initializer='zeros', activation='softmax')
    #         self.bit_model = module

    #     def call(self, images):
    #         out = self.bit_model(images)
    #         out = self.dense1(out)
    #         out = self.dropout(out)
    #         out = self.dense2(out)
    #         out = self.dropout(out)
    #         out = self.dense3(out)
    #         out = self.dropout(out)
    #         out = self.head(out)
    #         return out
    # lr=0.005 # start with this learning rate
    # model = MyBiTModel(num_classes=7, module=pre_trained_model)
    # 编译模型
    # model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy']) 
    # model.compile(optimizer=SGD(learning_rate=0.005, momentum=0.9),
    #           loss=CategoricalCrossentropy(label_smoothing=0.1),
    #           metrics=['accuracy'])

    # 保存搭建好的模型结构
    model.save("/data/mml/overlap_v2_datasets/animal/party_A/models/model_struct/VGG19.h5")
    # 设置检查点
    saved_dir_path = "/data/mml/overlap_v2_datasets/animal/party_A/models/model_weights"
    checkpointer = ModelCheckpoint(os.path.join(saved_dir_path, 'model_weight_{epoch:03d}_{val_accuracy:.4f}.h5'),
                                    verbose=1, 
                                    monitor='val_accuracy',
                                    save_weights_only=True, 
                                    save_best_only=True)
    # 设置学习率减小
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 10, verbose=1, factor=0.6, min_lr=0.000001)
    
    # 开始训练
    history = model.fit(train_batches,
                    # steps_per_epoch=reTrain_set_size/batch_size,
                    epochs=50, 
                    callbacks=[checkpointer,learning_rate_reduction], #[lr_scheduler, early_stopping, learning_rate_reduction], 
                    validation_data=val_batches,
                    # validation_steps = reValid_df_set_size/batch_size,
                    verbose = 1,
                    shuffle=True)


if __name__ == "__main__":
    start_train()