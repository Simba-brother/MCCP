'''
准备预训练模型 
'''
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# keras.models
from tensorflow.keras.models import Sequential, load_model
# keras.layers
from tensorflow.keras.layers import Layer, Dense, Conv2D, MaxPool2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Dropout
# keras.callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import Model
import sys
sys.path.append("/home/mml/workspace/model_reuse_v2/")
from utils import deleteIgnoreFile
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def getClasses(dir_path):
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    return classes_name_list

def start_train():
    # 加载数据
    train_csv_path = "/data/mml/overlap_v2_datasets/Fruit/party_A/dataset_split/train.csv"
    val_csv_path = "/data/mml/overlap_v2_datasets/Fruit/party_A/dataset_split/val.csv"
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # 声明出一个生成器
    train_gen =ImageDataGenerator(rescale = 1./255, rotation_range=20, width_shift_range=.2,
                                    height_shift_range=.2, zoom_range=.2, validation_split=0.2)  # 归一化
    val_gen =ImageDataGenerator(rescale = 1./255)  # 归一化

    # 获得batches
    classes = getClasses("/data/mml/overlap_v2_datasets/Fruit/party_A/dataset_split/train")
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
    pre_trained_model = load_model("/data/mml/overlap_v2_datasets/Fruit/party_A/models/standard_base_model/base_model_VGG19.h5")
    pre_trained_model.trainable=True
    # 构建模型
    model = Sequential([
        pre_trained_model,
        Flatten(),
        Dense(10 , activation='softmax')])
    lr=.0001 # start with this learning rate
    # 编译模型
    model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy']) 
    # 保存构建的模型结构
    model.save("/data/mml/overlap_v2_datasets/Fruit/party_A/models/model_struct/VGG19.h5")
    # 设置检查点
    saved_dir_path = "/data/mml/overlap_v2_datasets/Fruit/party_A/models/model_weights"
    checkpointer = ModelCheckpoint(os.path.join(saved_dir_path, 'model_weight_{epoch:03d}_{val_accuracy:.4f}.h5'),
                                    verbose=1, 
                                    monitor='val_accuracy',
                                    save_weights_only=True, 
                                    save_best_only=True)
    # 设置学习率减小
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 3, verbose=1,factor=0.6, min_lr=0.000001)
    # 设置一个停止训练的阈值
    early_stopping = EarlyStopping(patience=10)

        # 开始训练
    history = model.fit(train_batches,
                    # steps_per_epoch=reTrain_set_size/batch_size,
                    epochs=25, 
                    callbacks=[checkpointer, early_stopping], #[lr_scheduler, early_stopping, learning_rate_reduction], 
                    validation_data=val_batches,
                    # validation_steps = reValid_df_set_size/batch_size,
                    verbose = 1,
                    shuffle=True)

if __name__ == "__main__":
    start_train()