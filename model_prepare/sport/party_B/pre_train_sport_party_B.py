'''
准备预训练模型 
https://www.kaggle.com/code/omreekapon/sports-classification-using-vgg16-via-fine-tuning/notebook
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
sys.path.append("./")
from utils import deleteIgnoreFile
from tensorflow.keras.applications.efficientnet import EfficientNetB3
# keras application
from tensorflow.keras.applications import ResNet50, VGG19, Xception
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def configure_model(model_name):
  flatten=model_name.layers[-4]
  dropout1 = Dropout(0.3,name='Dropout1')
  dropout2 = Dropout(0.5,name='Dropout2')
  x=Dense(units=4096,activation='relu',name='FC1',kernel_regularizer='l2')(flatten.output)
  x = dropout1(x)
  x=Dense(units=4096,activation='relu',name='FC2',kernel_regularizer='l2')(x)
  x = dropout2(x)
  predictors = Dense(22,activation='softmax',name='Predictions')(x)
  final_model = Model(inputs=model_name.input, outputs=predictors)
  print(final_model.summary())
  return final_model

def getClasses(dir_path):
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    classes_name_list.sort()
    return classes_name_list

def start_train():
    # 加载数据
    train_csv_path = "/data/mml/overlap_v2_datasets/sport/party_B/dataset_split/train.csv"
    val_csv_path = "/data/mml/overlap_v2_datasets/sport/party_B/dataset_split/val.csv"
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # 声明出一个生成器
    train_gen =ImageDataGenerator(rescale=1./255,validation_split=0.2,zoom_range=0.5,horizontal_flip=True,rotation_range=40,vertical_flip=0.5,width_shift_range=0.3,height_shift_range=0.2,brightness_range=[0.2,1.0],fill_mode='nearest')  # 归一化
    val_gen =ImageDataGenerator(rescale=1./255)  # 归一化

    # 获得batches
    classes = getClasses("/data/mml/overlap_v2_datasets/sport/party_B/dataset_split/train")
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
                                                color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size)
                                
    print(train_batches.n)
    print(val_batches.n)
    print(test_batches.n)

    # 预训练模型
    pre_trained_model = load_model("/data/mml/overlap_v2_datasets/sport/party_B/models/standard_base_model/base_model_vgg16.h5")
    model = Sequential([
        pre_trained_model,
        # Flatten(),
        Dense(22 , activation='softmax')])
    # 构建模型
    # for layer in pre_trained_model.layers[:11]:
    #     layer.trainable=False
    # model = configure_model(pre_trained_model)
    lr=0.00003 # start with this learning rate
    model.compile(Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy']) 
    epochs = 100
    # 保存构建模型
    # model.save("/data/mml/overlap_v2_datasets/sport/party_B/models/model_struct/VGG16.h5")
    # 设置检查点
    saved_dir_path = "/data/mml/overlap_v2_datasets/sport/party_B/models/model_weights"
    checkpointer = ModelCheckpoint(os.path.join(saved_dir_path, 'model_weight_{epoch:03d}_{val_accuracy:.4f}.h5'),
                                    verbose=1, 
                                    monitor='val_accuracy',
                                    save_weights_only=True, 
                                    save_best_only=True)
    # 设置学习率减小
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 3, verbose=1,factor=0.6, min_lr=0.000001)
    # 设置一个停止训练的阈值
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10) # monitor='val_loss'

    # 开始训练
    history = model.fit_generator(train_batches,
                    steps_per_epoch=train_batches.samples/batch_size,
                    epochs=epochs, 
                    callbacks=[checkpointer, early_stopping], #[lr_scheduler, early_stopping, learning_rate_reduction], 
                    validation_data=val_batches,
                    validation_steps = val_batches.samples/batch_size,
                    verbose = 1,
                    shuffle=True)
    

if __name__ == "__main__":
    start_train()
