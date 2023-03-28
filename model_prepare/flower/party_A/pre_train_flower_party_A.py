'''
准备预训练模型
ref_code: https://www.kaggle.com/code/meetnagadia/flower-classification
'''
from tabnanny import verbose
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
# keras.models
from tensorflow.keras.models import Sequential, load_model
# keras.layers
from tensorflow.keras.layers import  Dense, MaxPool2D, Flatten, Dropout
# keras.callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from utils import deleteIgnoreFile
# 预训练模型
from tensorflow.keras.applications.vgg19 import VGG19
# 设置训练显卡
os.environ['CUDA_VISIBLE_DEVICES']='0'
from sklearn.utils import shuffle


def getClasses(dir_path):
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    classes_name_list.sort()  # 字典序
    return classes_name_list


def start_train():
    # 加载数据
    train_csv_path = "/data/mml/overlap_v2_datasets/flower/party_A/dataset_split/train.csv"
    val_csv_path = "/data/mml/overlap_v2_datasets/flower/party_A/dataset_split/val.csv"
    train_df = pd.read_csv(train_csv_path, dtype=str)
    val_df = pd.read_csv(val_csv_path, dtype=str)  # 从严格实验设置上它应该作为test集合

    # 声明出一个生成器
    # Keras ImageDataGenerator validation split not selected from shuffled dataset
    # https://stackoverflow.com/questions/62662194/keras-imagedatagenerator-validation-split-not-selected-from-shuffled-dataset
    train_gen =ImageDataGenerator(rescale = 1./255 , validation_split=0.2)  # 归一化, 同时要分化出验证集 
    test_gen =ImageDataGenerator(rescale = 1./255)  # 归一化
    # 获得batches
    classes = getClasses("/data/mml/overlap_v2_datasets/flower/party_A/dataset_split/train")
    train_df = train_df.sample(frac = 1)
    train_batches = train_gen.flow_from_dataframe(train_df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                subset="training",
                                                seed=42,
                                                x_col='file_path', y_col='label', 
                                                target_size=(224,224), class_mode='categorical',
                                                color_mode='rgb', classes=classes, shuffle=True, batch_size=32)

    val_batches = train_gen.flow_from_dataframe(train_df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                subset= "validation",
                                                seed=42,
                                                x_col='file_path', y_col='label', 
                                                target_size=(224,224), class_mode='categorical',
                                                color_mode='rgb', classes=classes,shuffle=True, batch_size=32)

    test_batches = test_gen.flow_from_dataframe(val_df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=(224,224), class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=False, batch_size=32)
    print(train_batches.n)
    print(val_batches.n)
    print(test_batches.n)

    # 准备模型
    # pre_trained_model = VGG19(input_shape=(224,224,3), include_top=False, weights="imagenet") 服务器没网络
    pre_trained_model  = load_model("/data/mml/overlap_v2_datasets/flower/party_A/models/standard_base_model/base_model_vgg19.h5")
    for layer in pre_trained_model.layers[:19]:
        layer.trainable = False
    model = Sequential([
        pre_trained_model,
        MaxPool2D((2,2) , strides = 2),
        Flatten(),
        Dense(5 , activation='softmax')])
    # 编译模型
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    ## optimizer="adam"
    model.compile(adam,loss="categorical_crossentropy",metrics="accuracy")
    model.save("/data/mml/overlap_v2_datasets/flower/party_A/models/model_struct/vgg19.h5")
    # 设置检查点
    saved_dir_path = "/data/mml/overlap_v2_datasets/flower/party_A/models/model_weights"
    checkpointer = ModelCheckpoint(os.path.join(saved_dir_path, 'model_weight_{epoch:03d}_{val_accuracy:.4f}.h5'),
                                    verbose=1, 
                                    monitor='val_accuracy',
                                    save_weights_only=True, 
                                    save_best_only=True)
    # 设置学习率减小
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 3, verbose=1,factor=0.6, min_lr=0.000001)

    # 开始训练
    history = model.fit(train_batches,
                    steps_per_epoch=train_batches.n//train_batches.batch_size,
                    epochs=25, 
                    callbacks=[checkpointer, learning_rate_reduction], #[lr_scheduler, early_stopping], 
                    validation_data=val_batches,
                    validation_steps = val_batches.n//val_batches.batch_size,
                    verbose = 1,
                    shuffle=True)
def start_eval():
    train_csv_path = "/data/mml/overlap_v2_datasets/flower/party_A/dataset_split/train.csv"
    val_csv_path = "/data/mml/overlap_v2_datasets/flower/party_A/dataset_split/val.csv"
    classes = getClasses("/data/mml/overlap_v2_datasets/flower/party_A/dataset_split/train")
    train_df = pd.read_csv(train_csv_path) 
    val_df = pd.read_csv(val_csv_path)  # 从严格实验设置上它应该作为test集合
    test_gen =ImageDataGenerator(rescale = 1./255)  # 归一化
    test_batches = test_gen.flow_from_dataframe(train_df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=(224,224), class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=False, batch_size=32)
    model_trained = load_model("/data/mml/overlap_v2_datasets/flower/party_A/models/model_struct/vgg19.h5")
    model_trained.load_weights("/data/mml/overlap_v2_datasets/flower/party_A/models/model_weights/model_weight_023_0.7364.h5")
    model_trained.evaluate_generator(generator=test_batches,steps=test_batches.n / 32, verbose = 1)

if __name__ == "__main__":
    start_train()
