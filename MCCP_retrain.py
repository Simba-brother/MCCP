import os
import logging
import setproctitle
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from utils import makedir_help
from sklearn.metrics import classification_report
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
        

class MCCP_Eval(object):

    def __init__(self, combin_model, dataset_name, df_merged:pd.DataFrame, df_test:pd.DataFrame):
        self.combin_model = combin_model
        self.dataset_name = dataset_name
        self.df_merged = df_merged
        self.df_test = df_test
    def eval(
            self,           
            target_size_A, 
            target_size_B, 
            generator_A_test,
            generator_B_test):
        
        root_dir = "/data2/mml/overlap_v2_datasets/"
        y_col = "label"  # importent!!!!!!
        classes = self.df_merged["label"].unique()
        classes = np.sort(classes).tolist()
        batch_size = 32
        test_batches_A = generator_A_test.flow_from_dataframe(self.df_test, 
                                                directory = root_dir, # 添加绝对路径前缀
                                                x_col='file_path', y_col=y_col, 
                                                target_size=target_size_A, class_mode='categorical',
                                                color_mode='rgb', classes = classes,
                                                shuffle=False, batch_size=batch_size,
                                                validate_filenames=False)
                                                                                                                # weather:rgb  150, 150

        test_batches_B = generator_B_test.flow_from_dataframe(self.df_test, 
                                                directory = root_dir, # 添加绝对路径前缀
                                                x_col='file_path', y_col=y_col, 
                                                target_size=target_size_B, class_mode='categorical',
                                                color_mode='rgb', classes = classes, 
                                                shuffle=False, batch_size=batch_size,
                                                validate_filenames=False)


        batches_test = generate_generator_multiple(test_batches_A, test_batches_B)
        
        res = self.combin_model.evaluate(batches_test, batch_size = batch_size, 
                                         verbose=1,steps = test_batches_A.n/batch_size, return_dict=True)
        # loss = res["loss"]
        # acc = res["accuracy"]
        return res  

    def predict_output(self,           
            target_size_A, 
            target_size_B, 
            generator_A_test,
            generator_B_test):
        root_dir = "/data2/mml/overlap_v2_datasets/"
        # importent!!!!!!
        y_col = "label"  
        classes = self.df_merged["label"].unique()
        classes = np.sort(classes).tolist()
        batch_size = 32
        test_batches_A = generator_A_test.flow_from_dataframe(self.df_test, 
                                                directory = root_dir, # 添加绝对路径前缀
                                                x_col='file_path', y_col=y_col, 
                                                target_size=target_size_A, class_mode='categorical',
                                                color_mode='rgb', classes = classes,
                                                shuffle=False, batch_size=batch_size,
                                                validate_filenames=False)
                                                                                                                # weather:rgb  150, 150

        test_batches_B = generator_B_test.flow_from_dataframe(self.df_test, 
                                                directory = root_dir, # 添加绝对路径前缀
                                                x_col='file_path', y_col=y_col, 
                                                target_size=target_size_B, class_mode='categorical',
                                                color_mode='rgb', classes = classes, 
                                                shuffle=False, batch_size=batch_size,
                                                validate_filenames=False)


        batches_test = generate_generator_multiple(test_batches_A, test_batches_B)
        
        output_prediction = self.combin_model.predict(batches_test, batch_size = batch_size, 
                                         verbose=1,steps = test_batches_A.n/batch_size)
        return output_prediction  

def app_MCCP_retrain_NoFangHui(config):
    root_dir = "/data2/mml/overlap_v2_datasets/"
    dataset_name =  config["dataset_name"]
    repeat_num = 10
    setproctitle.setproctitle(f"{dataset_name}|MCCP|retrain|NoFanghui")
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
        for repeat_i in range(repeat_num):
            print(f"repeat_i:{repeat_i}")
            dataset_retrain_csv_path = os.path.join(sample_rate_dir, f"sample_{repeat_i}.csv")
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
            save_dir = os.path.join(root_dir, dataset_name, "MCCP", "trained_weights_NoFangHui", str(int(sample_rate*100)))
            makedir_help(save_dir)
            save_file_name = f"weight_{repeat_i}.h5"
            save_file_path = os.path.join(save_dir, save_file_name)
            model.save_weights(save_file_path)
            print(f"save_file_path:{save_file_path}")
    print("app_MCCP_retrain_NoFangHui end")        


def app_MCCP_retrain_FangHui():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    config = animal_3_config
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    dataset_name =  config["dataset_name"]
    print(f"dataset_name:{dataset_name}")
    setproctitle.setproctitle(f"{dataset_name}|MCCP|train_FangHui")
    dataset_merged_df = pd.read_csv(config["merged_df_path"])
    dataset_test_df = pd.read_csv(config["merged_df_path"])
    sample_rate_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    retrain_dir = os.path.join("exp_data", dataset_name, "sampling", "percent", "random")
    for sample_rate in sample_rate_list:
        print(f"sample_rate:{sample_rate}")
        sample_rate_dir = os.path.join(retrain_dir, str(int(sample_rate*100))) 
        for repeat_num in range(5,10):
            print(f"repeat_num:{repeat_num}")
            dataset_retrain_csv_path = os.path.join(sample_rate_dir, f"sampled_{repeat_num}.csv")
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
            save_dir = os.path.join(root_dir, dataset_name, "MCCP", "trained_weights_FangHui", str(int(sample_rate*100)))
            makedir_help(save_dir)
            save_file_name = f"weight_{repeat_num}.h5"
            save_file_path = os.path.join(save_dir, save_file_name)
            model.save_weights(save_file_path)
            print(f"save_file_path:{save_file_path}")
    print("MCCP FangHui retraining end")   

def app_MCCP_eval_NoFangHui(config):
    sample_rate_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    # 定义出存储结果的数据结构
    '''
    ans = {
        0.01:[],
        0.03:[]
    }
    '''
    ans = {}
    for sample_rate in sample_rate_list:
        ans[sample_rate] = []
    root_dir = "/data2/mml/overlap_v2_datasets"
    repeat_num = 10
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|MCCP|eval|NoFangHui")
    print(f"dataset_name:{dataset_name}")
    combin_model = load_model(config["combination_model_path"])
    df_merged = pd.read_csv(config["merged_df_path"])
    test_dir = os.path.join("exp_data", dataset_name, "sampling", "percent", "random_split", "test")
    df_test = pd.read_csv(os.path.join(test_dir,"test.csv"))
    for sample_rate in sample_rate_list:
        print(f"sample_rate:{sample_rate}")
        for repeat_i in range(repeat_num):
            print(f"repeat_num:{repeat_i}")
            weight_path = os.path.join(root_dir, f"{dataset_name}", 
                                       "MCCP", "trained_weights_NoFangHui", 
                                       str(int(sample_rate*100)), f"weight_{repeat_i}.h5")
            
            adam = Adam(learning_rate=config["combiantion_lr"], beta_1=0.9, 
                        beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            combin_model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")
            combin_model.load_weights(weight_path)
            mccp_eval = MCCP_Eval(combin_model, dataset_name, df_merged, df_test)
            eval_res = mccp_eval.eval(
                target_size_A = config["target_size_A"],
                target_size_B = config["target_size_B"],
                generator_A_test = config["generator_A_test"],
                generator_B_test = config["generator_B_test"])
            acc = eval_res["accuracy"]
            ans[sample_rate].append(acc)
    
    save_dir = os.path.join(root_dir, dataset_name, "MCCP")
    save_file_name = f"eval_ans_NoFangHui.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("app_MCCP_eval_NoFangHui end")
    return ans

def app_MCCP_eval_FangHui():
    sample_rate_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    # 定义出存储结果的数据结构
    '''
    ans = {
        0.01:[],
        0.03:[]
    }
    '''
    ans = {}
    for sample_rate in sample_rate_list:
        ans[sample_rate] = []
    
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    config = animal_3_config
    root_dir = "/data2/mml/overlap_v2_datasets"
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|MCCP|eval|FangHui")
    print(f"dataset_name:{dataset_name}")
    combin_model = load_model(config["combination_model_path"])
    df_merged = pd.read_csv(config["merged_df_path"])
    df_test = pd.read_csv(config["merged_df_path"])
    for sample_rate in sample_rate_list:
        print(f"sample_rate:{sample_rate}")
        for repeat_num in range(10):
            print(f"repeat_num:{repeat_num}")
            weight_path = os.path.join(root_dir, f"{dataset_name}", "MCCP", "trained_weights_FangHui", str(int(sample_rate*100)), f"weight_{repeat_num}.h5")
            adam = Adam(learning_rate=config["combiantion_lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            combin_model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")
            combin_model.load_weights(weight_path)
            mccp_eval = MCCP_Eval(combin_model, dataset_name, df_merged, df_test)
            eval_res = mccp_eval.eval(
                target_size_A = config["target_size_A"],
                target_size_B = config["target_size_B"],
                generator_A_test = config["generator_A_test"],
                generator_B_test = config["generator_B_test"])
            acc = eval_res["accuracy"]
            ans[sample_rate].append(acc)
    
    save_dir = os.path.join(root_dir, dataset_name, "MCCP")
    save_file_name = f"eval_ans_FangHui.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("MCCP evaluation FangHui end")
    return ans

def app_MCCP_eval_Classes_FangHui():
    sample_rate_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    # 定义出存储结果的数据结构
    '''
    ans = {
        0.01:[],
        0.03:[]
    }
    '''
    ans = {}
    for sample_rate in sample_rate_list:
        ans[sample_rate] = []
    
    os.environ['CUDA_VISIBLE_DEVICES']='5'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    config = animal_3_config
    root_dir = "/data2/mml/overlap_v2_datasets"
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|MCCP|eval|FangHui")
    print(f"dataset_name:{dataset_name}")
    combin_model = load_model(config["combination_model_path"])
    df_merged = pd.read_csv(config["merged_df_path"])
    df_test = pd.read_csv(config["merged_df_path"])
    true_labels = df_test["label_globalIndex"].values
    for sample_rate in sample_rate_list:
        print(f"sample_rate:{sample_rate}")
        for repeat_num in range(10):
            print(f"repeat_num:{repeat_num}")
            weight_path = os.path.join(root_dir, f"{dataset_name}", "MCCP", "trained_weights_FangHui", str(int(sample_rate*100)), f"weight_{repeat_num}.h5")
            adam = Adam(learning_rate=config["combiantion_lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            combin_model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")
            combin_model.load_weights(weight_path)
            mccp_eval = MCCP_Eval(combin_model, dataset_name, df_merged, df_test)
            predict_output = mccp_eval.predict_output(
                target_size_A = config["target_size_A"],
                target_size_B = config["target_size_B"],
                generator_A_test = config["generator_A_test"],
                generator_B_test = config["generator_B_test"])
            predict_labels = np.argmax(predict_output, axis=1)
            report = classification_report(true_labels, predict_labels, output_dict=True)
            ans[sample_rate].append(report)
    
    save_dir = os.path.join(root_dir, dataset_name, "MCCP")
    save_file_name = f"eval_classes_FangHui.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("MCCP evaluation FangHui end")
    return ans

def app_MCCP_eval_Overlap_FangHui(config):
    sample_rate_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    # 定义出存储结果的数据结构
    '''
    ans = {
        0.01:[],
        0.03:[]
    }
    '''
    ans = {}
    for sample_rate in sample_rate_list:
        ans[sample_rate] = []
    root_dir = "/data2/mml/overlap_v2_datasets"
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|MCCP|eval_Overlap|FangHui")
    print(f"dataset_name:{dataset_name}")
    combin_model = load_model(config["combination_model_path"])
    df_merged = pd.read_csv(config["merged_df_path"])
    df_test = pd.read_csv(config["merged_overlap_df"])
    for sample_rate in sample_rate_list:
        print(f"sample_rate:{sample_rate}")
        for repeat_num in range(10):
            print(f"repeat_num:{repeat_num}")
            weight_path = os.path.join(root_dir, f"{dataset_name}", "MCCP", "trained_weights_FangHui", str(int(sample_rate*100)), f"weight_{repeat_num}.h5")
            adam = Adam(learning_rate=config["combiantion_lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            combin_model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")
            combin_model.load_weights(weight_path)
            mccp_eval = MCCP_Eval(combin_model, dataset_name, df_merged, df_test)
            eval_res = mccp_eval.eval(
                target_size_A = config["target_size_A"],
                target_size_B = config["target_size_B"],
                generator_A_test = config["generator_A_test"],
                generator_B_test = config["generator_B_test"])
            acc = eval_res["accuracy"]
            ans[sample_rate].append(acc)
    
    save_dir = os.path.join(root_dir, dataset_name, "MCCP")
    save_file_name = f"eval_ans_Overlap_FangHui.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("MCCP evaluation FangHui end")
    return ans  

def app_MCCP_eval_Unique_FangHui():
    sample_rate_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    # 定义出存储结果的数据结构
    '''
    ans = {
        0.01:[],
        0.03:[]
    }
    '''
    ans = {}
    for sample_rate in sample_rate_list:
        ans[sample_rate] = []
    
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    config = animal_2_config
    root_dir = "/data2/mml/overlap_v2_datasets"
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|MCCP|eval|FangHui")
    print(f"dataset_name:{dataset_name}")
    combin_model = load_model(config["combination_model_path"])
    df_merged = pd.read_csv(config["merged_df_path"])
    df_test = pd.read_csv(config["merged_unique_df"])
    for sample_rate in sample_rate_list:
        print(f"sample_rate:{sample_rate}")
        for repeat_num in range(10):
            print(f"repeat_num:{repeat_num}")
            weight_path = os.path.join(root_dir, f"{dataset_name}", "MCCP", "trained_weights_FangHui", str(int(sample_rate*100)), f"weight_{repeat_num}.h5")
            adam = Adam(learning_rate=config["combiantion_lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            combin_model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")
            combin_model.load_weights(weight_path)
            mccp_eval = MCCP_Eval(combin_model, dataset_name, df_merged, df_test)
            eval_res = mccp_eval.eval(
                target_size_A = config["target_size_A"],
                target_size_B = config["target_size_B"],
                generator_A_test = config["generator_A_test"],
                generator_B_test = config["generator_B_test"])
            acc = eval_res["accuracy"]
            ans[sample_rate].append(acc)
    
    save_dir = os.path.join(root_dir, dataset_name, "MCCP")
    save_file_name = f"eval_ans_Unique_FangHui.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("MCCP evaluation FangHui end")
    return ans  

def app_MCCP_eval_TrueFalse_FangHui():
    sample_rate_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    # 定义出存储结果的数据结构
    '''
    ans = {
        0.01:[],
        0.03:[]
    }
    '''
    ans = {}
    for sample_rate in sample_rate_list:
        ans[sample_rate] = []
    
    os.environ['CUDA_VISIBLE_DEVICES']='5'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    config = animal_2_config
    root_dir = "/data2/mml/overlap_v2_datasets"
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|MCCP|eval|FangHui")
    print(f"dataset_name:{dataset_name}")
    combin_model = load_model(config["combination_model_path"])
    df_merged = pd.read_csv(config["merged_df_path"])
    df_test = pd.read_csv(config["merged_df_path"])
    true_labels = df_test["label_globalIndex"].values
    for sample_rate in sample_rate_list:
        print(f"sample_rate:{sample_rate}")
        for repeat_num in range(10):
            print(f"repeat_num:{repeat_num}")
            weight_path = os.path.join(root_dir, f"{dataset_name}", "MCCP", "trained_weights_FangHui", str(int(sample_rate*100)), f"weight_{repeat_num}.h5")
            adam = Adam(learning_rate=config["combiantion_lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            combin_model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")
            combin_model.load_weights(weight_path)
            mccp_eval = MCCP_Eval(combin_model, dataset_name, df_merged, df_test)
            predict_output = mccp_eval.predict_output(
                target_size_A = config["target_size_A"],
                target_size_B = config["target_size_B"],
                generator_A_test = config["generator_A_test"],
                generator_B_test = config["generator_B_test"])
            predict_labels = np.argmax(predict_output, axis=1)
            trueOrFalse_list = np.equal(predict_labels, true_labels)
            ans[sample_rate].append(trueOrFalse_list)
    
    save_dir = os.path.join(root_dir, dataset_name, "MCCP")
    save_file_name = f"eval_TrueOrFalse_list_FangHui.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("MCCP evaluation FangHui end")
    return ans

def app_MCCP_init_overlap_merged_acc():    
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    config = fruit_config
    root_dir = "/data2/mml/overlap_v2_datasets"
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|MCCP|eval_init_overlap_merged_test|FangHui")
    print(f"dataset_name:{dataset_name}")
    combin_model = load_model(config["combination_model_path"])
    df_merged = pd.read_csv(config["merged_df_path"])
    df_test = pd.read_csv(config["merged_overlap_df"])
    adam = Adam(learning_rate=config["combiantion_lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    combin_model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")  
    mccp_eval = MCCP_Eval(combin_model, dataset_name, df_merged, df_test)
    eval_res = mccp_eval.eval(
        target_size_A = config["target_size_A"],
        target_size_B = config["target_size_B"],
        generator_A_test = config["generator_A_test"],
        generator_B_test = config["generator_B_test"])
    acc = eval_res["accuracy"]
    save_dir = os.path.join(root_dir, dataset_name, "MCCP")
    save_file_name = f"init_overlap_merged_test_acc.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(acc, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("app_MCCP_init_overlap_merged_acc end")
    return acc

def app_MCCP_init_unique_merged_acc():    
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    config = animal_3_config
    root_dir = "/data2/mml/overlap_v2_datasets"
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|MCCP|unique|init")
    print(f"dataset_name:{dataset_name}")
    combin_model = load_model(config["combination_model_path"])
    df_merged = pd.read_csv(config["merged_df_path"])
    df_test = pd.read_csv(config["merged_unique_df"])
    adam = Adam(learning_rate=config["combiantion_lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    combin_model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")  
    mccp_eval = MCCP_Eval(combin_model, dataset_name, df_merged, df_test)
    eval_res = mccp_eval.eval(
        target_size_A = config["target_size_A"],
        target_size_B = config["target_size_B"],
        generator_A_test = config["generator_A_test"],
        generator_B_test = config["generator_B_test"])
    acc = eval_res["accuracy"]
    save_dir = os.path.join(root_dir, dataset_name, "MCCP")
    save_file_name = f"init_unique_merged_test_acc.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(acc, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("app_MCCP_init_unique_merged_acc end")
    return acc

def app_MCCP_init_merged_acc():    
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    config = animal_3_config
    root_dir = "/data2/mml/overlap_v2_datasets"
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|MCCP|merged|init")
    print(f"dataset_name:{dataset_name}")
    combin_model = load_model(config["combination_model_path"])
    df_merged = pd.read_csv(config["merged_df_path"])
    adam = Adam(learning_rate=config["combiantion_lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    combin_model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")  
    mccp_eval = MCCP_Eval(combin_model, dataset_name, df_merged, df_merged)
    eval_res = mccp_eval.eval(
        target_size_A = config["target_size_A"],
        target_size_B = config["target_size_B"],
        generator_A_test = config["generator_A_test"],
        generator_B_test = config["generator_B_test"])
    acc = eval_res["accuracy"]
    save_dir = os.path.join(root_dir, dataset_name, "MCCP")
    save_file_name = f"init_merged_test_acc.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(acc, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("app_MCCP_init_merged_acc end")
    return acc

def app_MCCP_init_overlap_acc():    
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    config = animal_2_config
    root_dir = "/data2/mml/overlap_v2_datasets"
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|MCCP|merged|init")
    print(f"dataset_name:{dataset_name}")
    combin_model = load_model(config["combination_model_path"])
    df_merged = pd.read_csv(config["merged_df_path"])
    df_test = pd.read_csv(config["merged_overlap_df"])
    adam = Adam(learning_rate=config["combiantion_lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    combin_model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")  
    mccp_eval = MCCP_Eval(combin_model, dataset_name, df_merged, df_test)
    eval_res = mccp_eval.eval(
        target_size_A = config["target_size_A"],
        target_size_B = config["target_size_B"],
        generator_A_test = config["generator_A_test"],
        generator_B_test = config["generator_B_test"])
    acc = eval_res["accuracy"]
    save_dir = os.path.join(root_dir, dataset_name, "MCCP")
    save_file_name = f"init_overlap_test_acc.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(acc, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("app_MCCP_init_overlap_acc end")
    return acc

def app_MCCP_init_unique_acc():    
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    config = animal_2_config
    root_dir = "/data2/mml/overlap_v2_datasets"
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|MCCP|merged|init")
    print(f"dataset_name:{dataset_name}")
    combin_model = load_model(config["combination_model_path"])
    df_merged = pd.read_csv(config["merged_df_path"])
    df_test = pd.read_csv(config["merged_unique_df"])
    adam = Adam(learning_rate=config["combiantion_lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    combin_model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")  
    mccp_eval = MCCP_Eval(combin_model, dataset_name, df_merged, df_test)
    eval_res = mccp_eval.eval(
        target_size_A = config["target_size_A"],
        target_size_B = config["target_size_B"],
        generator_A_test = config["generator_A_test"],
        generator_B_test = config["generator_B_test"])
    acc = eval_res["accuracy"]
    save_dir = os.path.join(root_dir, dataset_name, "MCCP")
    save_file_name = f"init_unique_test_acc.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(acc, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("app_MCCP_init_overlap_acc end")
    return acc


def app_MCCP_init_merged_classes_acc():
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    config = animal_2_config
    root_dir = "/data2/mml/overlap_v2_datasets"
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|MCCP|merged|init")
    print(f"dataset_name:{dataset_name}")
    combin_model = load_model(config["combination_model_path"])
    df_merged = pd.read_csv(config["merged_df_path"])
    true_labels = df_merged["label_globalIndex"].values
    adam = Adam(learning_rate=config["combiantion_lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    combin_model.compile(optimizer=adam,loss="categorical_crossentropy",metrics="accuracy")  
    mccp_eval = MCCP_Eval(combin_model, dataset_name, df_merged, df_merged)
    predict_output = mccp_eval.predict_output(
        target_size_A = config["target_size_A"],
        target_size_B = config["target_size_B"],
        generator_A_test = config["generator_A_test"],
        generator_B_test = config["generator_B_test"])
    predict_labels = np.argmax(predict_output, axis=1)
    report = classification_report(true_labels, predict_labels, output_dict=True)
    save_dir = os.path.join(root_dir, dataset_name, "MCCP")
    save_file_name = f"init_merged_test_classes_acc.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(report, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("app_MCCP_init_merged_classes_acc end")
    return report



if __name__ == "__main__":
    # 设置GPU id
    os.environ['CUDA_VISIBLE_DEVICES']='7'
    # tf设置GPU内存分配
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    # 倒入数据集相关配置
    config = animal_3_config
    # NoFangHui MCCP train application
    # app_MCCP_retrain_NoFangHui(config)
    # NoFangHui MCCP eval application
    # app_MCCP_eval_NoFangHui(config)
    # app_MCCP_retrain_FangHui()
    app_MCCP_eval_FangHui()
    # app_MCCP_eval_Classes_FangHui()
    # app_MCCP_eval_TrueFalse_FangHui()
    # app_MCCP_init_overlap_merged_acc()
    # app_MCCP_init_unique_merged_acc()
    # app_MCCP_init_merged_acc()
    # app_MCCP_init_merged_classes_acc()
    # app_MCCP_eval_Overlap_FangHui(config)
    # app_MCCP_eval_Unique_FangHui()
    # app_MCCP_init_overlap_acc()
    # app_MCCP_init_unique_acc()
    pass