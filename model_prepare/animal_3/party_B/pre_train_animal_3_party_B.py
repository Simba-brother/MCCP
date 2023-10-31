'''
准备预训练模型 
refcode: https://www.kaggle.com/code/gpiosenka/animals-f1-score-100
'''
import pandas as pd
import os
import sys
import tensorflow as tf
print(tf.__version__)
sys.path.append("./")
from utils import deleteIgnoreFile

# keras.models
from tensorflow.keras.models import Sequential, load_model,Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# keras.layers
from tensorflow.keras.layers import Layer, Dense, Conv2D, MaxPool2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Dropout
# keras.callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adamax

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def getClasses(dir_path):
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    return classes_name_list

def start_train():
    # 加载数据
    train_csv_path = "/data/mml/overlap_v2_datasets/animal_3/party_B/dataset_split/train.csv"
    val_csv_path = "/data/mml/overlap_v2_datasets/animal_3/party_B/dataset_split/val.csv"
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # 声明出一个生成器
    train_gen =ImageDataGenerator(
                                horizontal_flip=True,  rotation_range=20, width_shift_range=.2,
                                height_shift_range=.2, zoom_range=.2,
                                validation_split=0.2)
    val_gen =ImageDataGenerator() 

    # 获得batches
    classes = getClasses("/data/mml/overlap_v2_datasets/animal_3/party_B/dataset_split/train")
    train_df = train_df.sample(frac = 1)
    target_size = (224,224)
    batch_size = 32
    train_batches = train_gen.flow_from_dataframe(train_df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                subset="training",
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size, class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=True, batch_size=batch_size)

    val_batches = train_gen.flow_from_dataframe(train_df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                subset= "validation",
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size, class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=True, batch_size=batch_size)
    test_batches = val_gen.flow_from_dataframe(val_df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size, class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=False, batch_size=32)

    print(train_batches.n)
    print(val_batches.n)
    print(test_batches.n)

    base_model = load_model("/data/mml/overlap_v2_datasets/animal_3/party_B/models/standard_base_model/base_model_EfficientNetB3_224_224_poolingMax.h5")
    base_model.trainable=True
    x=base_model.output
    x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
    x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                    bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
    x=Dropout(rate=.4, seed=123)(x)  
    class_count = 6   
    output=Dense(class_count, activation='softmax')(x)
    model=Model(inputs=base_model.input, outputs=output)
    model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy']) 
    # 保存搭建好的模型结构
    model.save("/data/mml/overlap_v2_datasets/animal_3/party_B/models/model_struct/EfficientNetB3_224_224_poolingMax.h5")
    # 设置检查点
    saved_dir_path = "/data/mml/overlap_v2_datasets/animal_3/party_B/models/model_weights"
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
                    epochs=40, 
                    callbacks=[checkpointer,learning_rate_reduction], #[lr_scheduler, early_stopping, learning_rate_reduction], 
                    validation_data=val_batches,
                    # validation_steps = reValid_df_set_size/batch_size,
                    verbose = 1,
                    shuffle=True)    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.show()
if __name__ == "__main__":
    start_train()
