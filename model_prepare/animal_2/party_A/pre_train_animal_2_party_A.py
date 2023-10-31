'''
准备预训练模型 
refcode: https://www.kaggle.com/code/rizkiprof/image-classification-cnn-with-resnet
'''
import pandas as pd
import os
import sys
sys.path.append("./")
from utils import deleteIgnoreFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# keras.models
from tensorflow.keras.models import Sequential, load_model
# keras.layers
from tensorflow.keras.layers import Layer, Dense, Conv2D, MaxPool2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Dropout
# keras.callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def getClasses(dir_path):
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    return classes_name_list

def start_train():
    # 加载数据
    train_csv_path = "/data/mml/overlap_v2_datasets/animal_2/party_A/data_split/train.csv"
    val_csv_path = "/data/mml/overlap_v2_datasets/animal_2/party_A/data_split/val.csv"
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # 声明出一个生成器
    train_gen =ImageDataGenerator(fill_mode = 'nearest',validation_split=0.2)
    val_gen =ImageDataGenerator() 

    # 获得batches
    classes = getClasses("/data/mml/overlap_v2_datasets/animal_2/party_A/data_split/train")
    train_df = train_df.sample(frac = 1)
    target_size = (108,108)
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
    base_model = load_model("/data/mml/overlap_v2_datasets/animal_2/party_A/models/standard_base_model/base_model_ResNet50_108_108.h5")
    model = tf.keras.models.Sequential([base_model])
    for layer in model.layers:
        layer.trainable = False
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    # 保存搭建好的模型结构
    model.save("/data/mml/overlap_v2_datasets/animal_2/party_A/models/model_struct/ResNet50.h5")
    # 设置检查点
    saved_dir_path = "/data/mml/overlap_v2_datasets/animal_2/party_A/models/model_weights"
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
                    epochs=25, 
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
