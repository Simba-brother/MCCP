import os
import logging

import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from utils import makedir_help
from DataSetConfig import car_body_style_config,flower_2_config,food_config,fruit_config, sport_config, weather_config, animal_config, animal_2_config, animal_3_config


def generate_generator_multiple(batches_A, batches_B):
    '''
    将连个模型的输入bath 同时返回
    '''
    while True:
        X1i = batches_A.next()
        X2i = batches_B.next()
        yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label

class Retrain(object):

    def __init__(self, combin_model, dataset_name, dataset_retrain:pd.DataFrame, dataset_merged:pd.DataFrame, dataset_test:pd.DataFrame):
        self.combin_model = combin_model
        self.dataset_name = dataset_name
        self.dataset_retrain = dataset_retrain
        self.dataset_merged = dataset_merged
        self.dataset_test = dataset_test


    def train(self, 
              epochs, 
              batch_size, 
              lr, 
              target_size_A, 
              target_size_B, 
              generator_A, 
              generator_B,
              generator_A_test,
              generator_B_test):
        
        model = self.combin_model
        df_retrain = self.dataset_retrain
        df_merged = self.dataset_merged
        classes = df_merged["label"].unique()
        classes = np.sort(classes).tolist()
        df_test = self.dataset_test
        y_col = "label"  # importent!!!!!!
        root_dir = "/data2/mml/overlap_v2_datasets/"
        batches_A = generator_A.flow_from_dataframe(df_retrain, 
                                                    directory = root_dir, # 添加绝对路径前缀
                                                    # subset="training",
                                                    seed=666,
                                                    x_col='file_path', y_col=y_col, 
                                                    target_size=target_size_A, class_mode='categorical',
                                                    color_mode='rgb', classes=classes, 
                                                    shuffle=False, batch_size=batch_size,
                                                    validate_filenames=False)
        batches_B = generator_B.flow_from_dataframe(df_retrain, 
                                                    directory = root_dir, 
                                                    # subset="training",
                                                    seed=666,
                                                    x_col='file_path', y_col=y_col, 
                                                    target_size=target_size_B, class_mode='categorical',
                                                    color_mode='rgb', classes=classes, 
                                                    shuffle=False, batch_size=batch_size,
                                                    validate_filenames=False)

        batches_train = generate_generator_multiple(batches_A, batches_B)

        batches_A_test = generator_A_test.flow_from_dataframe(df_test, 
                                                            directory = root_dir, # 添加绝对路径前缀
                                                            # subset="training",
                                                            seed=666,
                                                            x_col='file_path', y_col= y_col, 
                                                            target_size=target_size_A, class_mode='categorical',
                                                            color_mode='rgb', classes=classes, 
                                                            shuffle=False, batch_size=batch_size,
                                                            validate_filenames=False)
        batches_B_test = generator_B_test.flow_from_dataframe(df_test, 
                                                        directory = root_dir, # 添加绝对路径前缀
                                                        # subset="training",
                                                        seed=666,
                                                        x_col='file_path', y_col= y_col, 
                                                        target_size=target_size_B, class_mode='categorical',
                                                        color_mode='rgb', classes=classes, 
                                                        shuffle=False, batch_size=batch_size,
                                                        validate_filenames=False)

        batches_test = generate_generator_multiple(batches_A_test, batches_B_test)

        # 模型编译 
        adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")
        # 开始训练
        history = model.fit(batches_train,
                        steps_per_epoch=df_retrain.shape[0]//batch_size,
                        epochs = epochs,
                        # callbacks=[checkpointer, learning_rate_reduction], #[lr_scheduler, early_stopping], 
                        validation_data=batches_test,
                        validation_steps = df_test.shape[0]//batch_size,
                        verbose = 1,
                        shuffle=False)
        return model
        

def app_MCCP_retrain():
    os.environ['CUDA_VISIBLE_DEVICES']='7'
    config = animal_3_config
    dataset_name =  config["dataset_name"]
    # log_save_dir = os.path.join('log',f"{dataset_name}")
    # makedir_help(log_save_dir)
    # log_file_name = "MCCP.log"
    # log_file_path = os.path.join(log_save_dir, log_file_name)
    # logging.basicConfig(filename=log_file_path, filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=logging.INFO)
    print(f"dataset_name:{dataset_name}")
    dataset_merged_df = pd.read_csv(config["merged_df_path"])
    test_dir = os.path.join("exp_data", dataset_name, "sampling", "percent", "random_split", "test")
    dataset_test_df = pd.read_csv(os.path.join(test_dir,"test.csv"))
    sample_rate_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    retrain_dir = os.path.join("exp_data", dataset_name, "sampling", "percent", "random_split", "train")
    for sample_rate in sample_rate_list:
        print(f"sample_rate:{sample_rate}")
        sample_rate_dir = os.path.join(retrain_dir, str(int(sample_rate*100))) 
        for repeat_num in range(5):
            print(f"repeat_num:{repeat_num}")
            dataset_retrain_csv_path = os.path.join(sample_rate_dir, f"sample_{repeat_num}.csv")
            dataset_retrain_df = pd.read_csv(dataset_retrain_csv_path)
            combin_model = load_model(config["combination_model_path"])
            retrain = Retrain(combin_model, dataset_name, dataset_retrain_df, dataset_merged_df, dataset_test_df)
            model = retrain.train(
                epochs=5, 
                batch_size=8, 
                lr=config["combiantion_lr"], 
                target_size_A=config["target_size_A"], 
                target_size_B=config["target_size_B"], 
                generator_A=config["generator_A"], 
                generator_B=config["generator_B"],
                generator_A_test=config["generator_A_test"],
                generator_B_test=config["generator_B_test"]
                )
            root_dir = "/data2/mml/overlap_v2_datasets/"
            save_dir = os.path.join(root_dir, dataset_name, "MCCP", "trained_weights", str(int(sample_rate*100)))
            makedir_help(save_dir)
            save_file_name = f"weight_{repeat_num}.h5"
            save_file_path = os.path.join(save_dir, save_file_name)
            model.save_weights(save_file_path)
            print(f"save_file_path:{save_file_path}")
    print("MCCP retraining end")        

if __name__ == "__main__":
   app_MCCP_retrain()