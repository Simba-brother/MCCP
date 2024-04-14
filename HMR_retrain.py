import os
import joblib
import setproctitle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import load_model,Model
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adamax
from DataSetConfig import car_body_style_config,flower_2_config,food_config, fruit_config,sport_config,weather_config,animal_config,animal_2_config,animal_3_config
from utils import deleteIgnoreFile,makedir_help

def getClasses(dir_path):
    '''
    得到数据集目录的class_name_list
    '''
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    classes_name_list.sort()
    return classes_name_list

def predict_proba(i, models_pool, generator_test_list, target_size_list, df, batch_size, root_dir):
    '''
    i方的对df的预测概率
    '''
    model = models_pool[i]
    # hasReserved = hasReceived_pool[i] 
    generator = generator_test_list[i]
    target_size = target_size_list[i]
    batches = generator.flow_from_dataframe(
        df, 
        directory = root_dir, # 添加绝对路径前缀
        # subset="training",
        seed=666,
        x_col='file_path', y_col="label", 
        target_size=target_size, class_mode='categorical', # one-hot
        color_mode='rgb', classes=None, 
        shuffle=False, batch_size=batch_size,
        validate_filenames=False)
    proba = model.predict(
        batches,    
        batch_size=None,
        verbose="auto",
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False)
    return proba[:, :-1]
    
def local_to_global(i,proba_local, all_class_nums, localToGlobal_mapping):
    '''
    i方的proba => globa proba
    '''
    predict_value = np.zeros((proba_local.shape[0], all_class_nums))
    localToGlobal_dic = localToGlobal_mapping[i]
    mapping = []
    for key, value in localToGlobal_dic.items():
        mapping.append(value)
    predict_value[:, mapping] = proba_local
    return predict_value

def add_reserve_class(model):
    '''
    给模型增加一个保留class神经元
    '''
    # copy the original weights, to keep the predicting function same
    weights_bak = model.layers[-1].get_weights()
    num_classes = model.layers[-1].output_shape[-1]
    # model.pop()
    # model.add(Dense(num_classes + 1, activation='softmax'))
    # cut 最后输出
    model_cut = keras.Model(inputs = model.input, outputs = model.get_layer(index = -2).output)
    # 声明出一个新输出层
    new_output_layer = Dense(num_classes + 1, activation='softmax', name="new_output_layer")(model_cut.output)
    # 重新连上新输出层
    model = Model(inputs = model_cut.input, outputs = new_output_layer)
    # model.summary()
    weights_new = model.layers[-1].get_weights()
    weights_new[0][:, :-1] = weights_bak[0]
    weights_new[1][:-1] = weights_bak[1]

    # use the average weight to init the last. This suppress its output, while keeping performance.
    weights_new[0][:, -1] = np.mean(weights_bak[0], axis=1)
    weights_new[1][-1] = np.mean(weights_bak[1])

    model.layers[-1].set_weights(weights_new)

    # model.compile(loss=categorical_crossentropy,
    #             optimizer=Adam(learning_rate=1e-4),
    #             metrics=['accuracy'])
    return model

def load_models_pool(config, lr=1e-5):
    # 加载模型
    model_A = load_model(config["model_A_struct_path"])
    if not config["model_A_weight_path"] is None:
        model_A.load_weights(config["model_A_weight_path"])

    model_B = load_model(config["model_B_struct_path"])
    if not config["model_B_weight_path"] is None:
        model_B.load_weights(config["model_B_weight_path"])
    model_A = add_reserve_class(model_A)
    model_B = add_reserve_class(model_B)
    model_A.compile(loss=categorical_crossentropy,
                optimizer=Adamax(learning_rate=lr),
                metrics=['accuracy'])
    model_B.compile(loss=categorical_crossentropy,
                optimizer=Adamax(learning_rate=lr),
                metrics=['accuracy'])
    # # 各方模型池
    # models_pool = [model_A, model_B]
    # # 模型池中的模型是否具备保留类功能
    # hasReceived_pool = [True, True]
    return model_A,model_B

class HMR_Retrain(object):
    def __init__(self, model_A,model_B,df_retrain):
        self.model_A = model_A
        self.model_B = model_B
        self.df_retrain = df_retrain
        self.root_dir = "/data2/mml/overlap_v2_datasets/"

    def train_A(self, epochs, batch_size, class_name_list, generator_train, target_size):
        model = self.model_A
        df_retrain = self.df_retrain.copy()
        df_retrain.loc[(df_retrain["source"]!=1)&(df_retrain["is_overlap"]==0), "label"] = "zzz"
        ex_classes= []
        ex_classes.extend(class_name_list)
        ex_classes.append("zzz")  # importent!!
        batches = generator_train.flow_from_dataframe(
            df_retrain, 
            directory = self.root_dir, # 添加绝对路径前缀
            seed=42,
            x_col='file_path', y_col="label", 
            target_size=target_size, 
            class_mode='categorical', # one-hot
            color_mode='rgb', classes=ex_classes, 
            shuffle=True, batch_size=batch_size,
            validate_filenames=False)
        # 开始训练
        history = model.fit(
            batches,
            steps_per_epoch=df_retrain.shape[0]//batch_size,
            epochs = epochs,
            # callbacks=[checkpointer, learning_rate_reduction], #[lr_scheduler, early_stopping], 
            # validation_data=batches_test,
            # validation_steps = merged_df.shape[0]//batch_size,
            verbose = 1,
            shuffle=True)
        return model

    def train_B(self, epochs, batch_size, class_name_list, generator_train, target_size):
        model = self.model_B
        df_retrain = self.df_retrain.copy()
        df_retrain.loc[(df_retrain["source"]!=2)&(df_retrain["is_overlap"]==0), "label"] = "zzz"
        ex_classes= []
        ex_classes.extend(class_name_list)
        ex_classes.append("zzz")  # importent!!
        batches = generator_train.flow_from_dataframe(
            df_retrain, 
            directory = self.root_dir, # 添加绝对路径前缀
            seed=42,
            x_col='file_path', y_col="label", 
            target_size=target_size, 
            class_mode='categorical', # one-hot
            color_mode='rgb', classes=ex_classes, 
            shuffle=True, batch_size=batch_size,
            validate_filenames=False)
        # 开始训练
        history = model.fit(
            batches,
            steps_per_epoch=df_retrain.shape[0]//batch_size,
            epochs = epochs,
            # callbacks=[checkpointer, learning_rate_reduction], #[lr_scheduler, early_stopping], 
            # validation_data=batches_test,
            # validation_steps = merged_df.shape[0]//batch_size,
            verbose = 1,
            shuffle=True)
        return model


class HMR_Eval(object):
    def __init__(self, model_A_extend,model_B_extend,df_test):
        self.model_A_extend = model_A_extend
        self.model_B_extend = model_B_extend
        self.df_test = df_test


    def eval(self, generator_A_test, generator_B_test, target_size_A, target_size_B, 
             all_class_nums, local_to_global_party_A,local_to_global_party_B):
        models_pool = [self.model_A_extend, self.model_B_extend]
        generator_test_list = [generator_A_test, generator_B_test]
        target_size_list = [target_size_A, target_size_B]
        localToGlobal_mapping = [local_to_global_party_A,local_to_global_party_B]
        batch_size = 32
        n = self.df_test.shape[0]
        root_dir = "/data2/mml/overlap_v2_datasets/"
        predict_value = np.zeros((len(models_pool), n, all_class_nums))
        for i,model in enumerate(models_pool):
            proba_local =predict_proba(i, models_pool, generator_test_list, target_size_list, self.df_test, batch_size, root_dir)
            proba_global = local_to_global(i, proba_local, all_class_nums, localToGlobal_mapping)
            predict_value[i, :, :] = proba_global
        proba_vector = np.max(predict_value, axis=0)
        predict_global_idx_array = np.argmax(proba_vector,axis=1)
        true_global_idx_array = np.array(self.df_test["label_globalIndex"])
        acc = np.sum(predict_global_idx_array == true_global_idx_array) / self.df_test.shape[0]
        return acc
    def predict_output(self, generator_A_test, generator_B_test, target_size_A, target_size_B, 
             all_class_nums, local_to_global_party_A,local_to_global_party_B):
        models_pool = [self.model_A_extend, self.model_B_extend]
        generator_test_list = [generator_A_test, generator_B_test]
        target_size_list = [target_size_A, target_size_B]
        localToGlobal_mapping = [local_to_global_party_A,local_to_global_party_B]
        batch_size = 32
        n = self.df_test.shape[0]
        root_dir = "/data2/mml/overlap_v2_datasets/"
        predict_value = np.zeros((len(models_pool), n, all_class_nums))
        for i,model in enumerate(models_pool):
            proba_local =predict_proba(i, models_pool, generator_test_list, target_size_list, self.df_test, batch_size, root_dir)
            proba_global = local_to_global(i, proba_local, all_class_nums, localToGlobal_mapping)
            predict_value[i, :, :] = proba_global
        proba_vector = np.max(predict_value, axis=0)
        # output_predict = np.argmax(proba_vector,axis=1)
        return proba_vector

def app_HMR_retrain():
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    root_dir = "/data2/mml/overlap_v2_datasets/"
    config = car_body_style_config
    dataset_name = config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|HMR|retrain")
    class_name_list_A = getClasses(config["dataset_A_train_path"]) # sorted
    class_name_list_B = getClasses(config["dataset_B_train_path"]) # sorted
    generator_A = config["generator_A"]
    generator_B = config["generator_B"]
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    train_dir = f"exp_data/{dataset_name}/sampling/percent/random_split/train"
    sample_rate_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    for sample_rate in sample_rate_list:
        sample_rate_dir = os.path.join(train_dir,str(int(sample_rate*100))) 
        for repeat_num in range(5):
            df_retrain = pd.read_csv(os.path.join(sample_rate_dir, f"sample_{repeat_num}.csv"))
            model_A_extend, model_B_extend = load_models_pool(config) # 添加新类和编译
            hmr_retrain = HMR_Retrain(model_A_extend,model_B_extend,df_retrain)
            model_A_extend_retrained = hmr_retrain.train_A(
                epochs=5, 
                batch_size=5, 
                class_name_list=class_name_list_A, 
                generator_train = generator_A, 
                target_size=target_size_A)
            model_B_extend_retrained = hmr_retrain.train_B(
                epochs=5, 
                batch_size=5, 
                class_name_list=class_name_list_B, 
                generator_train = generator_B, 
                target_size=target_size_B)
            save_dir = os.path.join(root_dir, dataset_name, "HMR", "trained_weights", str(int(sample_rate*100)))
            makedir_help(save_dir)
            save_file_name = f"model_A_weight_{repeat_num}.h5"
            save_file_path = os.path.join(save_dir, save_file_name)
            model_A_extend_retrained.save_weights(save_file_path)
            print(f"save_file_path:{save_file_path}")
            save_file_name = f"model_B_weight_{repeat_num}.h5"
            save_file_path = os.path.join(save_dir, save_file_name)
            model_B_extend_retrained.save_weights(save_file_path)
            print(f"save_file_path:{save_file_path}")
    print("app HMR retraining end")            
            
def app_HMR_retrain_FangHui():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    root_dir = "/data2/mml/overlap_v2_datasets/"
    config = animal_2_config
    dataset_name = config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|HMR|retrain|FangHui")
    class_name_list_A = getClasses(config["dataset_A_train_path"]) # sorted
    class_name_list_B = getClasses(config["dataset_B_train_path"]) # sorted
    generator_A = config["generator_A"]
    generator_B = config["generator_B"]
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    train_dir = f"exp_data/{dataset_name}/sampling/percent/random"
    sample_rate_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    for sample_rate in sample_rate_list:
        sample_rate_dir = os.path.join(train_dir,str(int(sample_rate*100))) 
        for repeat_num in range(10):
            df_retrain = pd.read_csv(os.path.join(sample_rate_dir, f"sampled_{repeat_num}.csv"))
            model_A_extend, model_B_extend = load_models_pool(config) # 添加新类和编译
            hmr_retrain = HMR_Retrain(model_A_extend,model_B_extend,df_retrain)
            model_A_extend_retrained = hmr_retrain.train_A(
                epochs=5, 
                batch_size=5, 
                class_name_list=class_name_list_A, 
                generator_train = generator_A, 
                target_size=target_size_A)
            model_B_extend_retrained = hmr_retrain.train_B(
                epochs=5, 
                batch_size=5, 
                class_name_list=class_name_list_B, 
                generator_train = generator_B, 
                target_size=target_size_B)
            save_dir = os.path.join(root_dir, dataset_name, "HMR", "trained_weights_FangHui_seed42", str(int(sample_rate*100)))
            makedir_help(save_dir)
            save_file_name = f"model_A_weight_{repeat_num}.h5"
            save_file_path = os.path.join(save_dir, save_file_name)
            model_A_extend_retrained.save_weights(save_file_path)
            print(f"save_file_path:{save_file_path}")
            save_file_name = f"model_B_weight_{repeat_num}.h5"
            save_file_path = os.path.join(save_dir, save_file_name)
            model_B_extend_retrained.save_weights(save_file_path)
            print(f"save_file_path:{save_file_path}")
    print("app HMR retraining FangHui end")

def app_HMR_eval():
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
    
    os.environ['CUDA_VISIBLE_DEVICES']='7'
    root_dir = "/data2/mml/overlap_v2_datasets/"
    config = sport_config
    dataset_name =  config["dataset_name"]
    model_A_extend, model_B_extend = load_models_pool(config) # 构建和编译
    setproctitle.setproctitle(f"{dataset_name}|HMR|eval")
    test_dir = f"exp_data/{dataset_name}/sampling/percent/random_split/test"
    df_test = pd.read_csv(os.path.join(test_dir, "test.csv"))
    class_name_list_A = getClasses(config["dataset_A_train_path"]) # sorted
    class_name_list_B = getClasses(config["dataset_B_train_path"]) # sorted
    all_class_name_list = list(set(class_name_list_A+class_name_list_B))
    all_class_name_list.sort()
    # 总分类数
    all_class_nums = len(all_class_name_list)
    generator_A_test = config["generator_A_test"]
    generator_B_test = config["generator_B_test"]
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    # 双方的local to global
    local_to_global_party_A = joblib.load(config["local_to_global_party_A_path"])
    local_to_global_party_B = joblib.load(config["local_to_global_party_B_path"])
    for sample_rate in sample_rate_list:
        for repeat_num in range(5):
            weight_A_path = os.path.join(root_dir, f"{dataset_name}", "HMR", "trained_weights", str(int(sample_rate*100)), f"model_A_weight_{repeat_num}.h5")
            weight_B_path = os.path.join(root_dir, f"{dataset_name}", "HMR", "trained_weights", str(int(sample_rate*100)), f"model_B_weight_{repeat_num}.h5")
            model_A_extend.load_weights(weight_A_path)
            model_B_extend.load_weights(weight_B_path)
            hmr_eval = HMR_Eval(model_A_extend, model_B_extend, df_test)
            acc = hmr_eval.eval(
                generator_A_test, 
                generator_B_test, 
                target_size_A, 
                target_size_B, 
                all_class_nums, 
                local_to_global_party_A,
                local_to_global_party_B)
            ans[sample_rate].append(acc)
    save_dir = os.path.join(root_dir, dataset_name, "HMR")
    save_file_name = f"eval_ans.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("HMR evaluation end")
    return ans

def app_HMR_eval_FangHui():
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
    root_dir = "/data2/mml/overlap_v2_datasets/"
    config = animal_2_config
    dataset_name =  config["dataset_name"]
    model_A_extend, model_B_extend = load_models_pool(config) # 构建和编译
    setproctitle.setproctitle(f"{dataset_name}|HMR|eval|FangHui")
    df_test = pd.read_csv(config["merged_df_path"])
    class_name_list_A = getClasses(config["dataset_A_train_path"]) # sorted
    class_name_list_B = getClasses(config["dataset_B_train_path"]) # sorted
    all_class_name_list = list(set(class_name_list_A+class_name_list_B))
    all_class_name_list.sort()
    # 总分类数
    all_class_nums = len(all_class_name_list)
    generator_A_test = config["generator_A_test"]
    generator_B_test = config["generator_B_test"]
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    # 双方的local to global
    local_to_global_party_A = joblib.load(config["local_to_global_party_A_path"])
    local_to_global_party_B = joblib.load(config["local_to_global_party_B_path"])
    for sample_rate in sample_rate_list:
        for repeat_num in range(10):
            weight_A_path = os.path.join(root_dir, f"{dataset_name}", "HMR", "trained_weights_FangHui_seed42", str(int(sample_rate*100)), f"model_A_weight_{repeat_num}.h5")
            weight_B_path = os.path.join(root_dir, f"{dataset_name}", "HMR", "trained_weights_FangHui_seed42", str(int(sample_rate*100)), f"model_B_weight_{repeat_num}.h5")
            model_A_extend.load_weights(weight_A_path)
            model_B_extend.load_weights(weight_B_path)
            hmr_eval = HMR_Eval(model_A_extend, model_B_extend, df_test)
            acc = hmr_eval.eval(
                generator_A_test, 
                generator_B_test, 
                target_size_A, 
                target_size_B, 
                all_class_nums, 
                local_to_global_party_A,
                local_to_global_party_B)
            ans[sample_rate].append(acc)
    save_dir = os.path.join(root_dir, dataset_name, "HMR")
    save_file_name = f"eval_ans_FangHui_seed42.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("HMR evaluation FangHui end")
    return ans

def app_HMR_eval_classes_FangHui():
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
    
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    root_dir = "/data2/mml/overlap_v2_datasets/"
    config = animal_3_config
    dataset_name =  config["dataset_name"]
    model_A_extend, model_B_extend = load_models_pool(config) # 构建和编译
    setproctitle.setproctitle(f"{dataset_name}|HMR|eval|FangHui")
    df_test = pd.read_csv(config["merged_df_path"])
    true_labels = df_test["label_globalIndex"].values
    class_name_list_A = getClasses(config["dataset_A_train_path"]) # sorted
    class_name_list_B = getClasses(config["dataset_B_train_path"]) # sorted
    all_class_name_list = list(set(class_name_list_A+class_name_list_B))
    all_class_name_list.sort()
    # 总分类数
    all_class_nums = len(all_class_name_list)
    generator_A_test = config["generator_A_test"]
    generator_B_test = config["generator_B_test"]
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    # 双方的local to global
    local_to_global_party_A = joblib.load(config["local_to_global_party_A_path"])
    local_to_global_party_B = joblib.load(config["local_to_global_party_B_path"])
    for sample_rate in sample_rate_list:
        for repeat_num in range(10):
            weight_A_path = os.path.join(root_dir, f"{dataset_name}", "HMR", "trained_weights_FangHui", str(int(sample_rate*100)), f"model_A_weight_{repeat_num}.h5")
            weight_B_path = os.path.join(root_dir, f"{dataset_name}", "HMR", "trained_weights_FangHui", str(int(sample_rate*100)), f"model_B_weight_{repeat_num}.h5")
            model_A_extend.load_weights(weight_A_path)
            model_B_extend.load_weights(weight_B_path)
            hmr_eval = HMR_Eval(model_A_extend, model_B_extend, df_test)
            output_predict = hmr_eval.predict_output(
                generator_A_test, 
                generator_B_test, 
                target_size_A, 
                target_size_B, 
                all_class_nums, 
                local_to_global_party_A,
                local_to_global_party_B)
            predict_labels = np.argmax(output_predict, axis = 1)
            report = classification_report(true_labels, predict_labels, output_dict=True)
            ans[sample_rate].append(report)
    save_dir = os.path.join(root_dir, dataset_name, "HMR")
    save_file_name = f"eval_classes_FangHui.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("HMR evaluation FangHui end")
    return ans

def app_HMR_eval_TrueFalse_FangHui():
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
    
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    root_dir = "/data2/mml/overlap_v2_datasets/"
    config = animal_2_config
    dataset_name =  config["dataset_name"]
    model_A_extend, model_B_extend = load_models_pool(config) # 构建和编译
    setproctitle.setproctitle(f"{dataset_name}|HMR|eval|FangHui")
    df_test = pd.read_csv(config["merged_df_path"])
    true_labels = df_test["label_globalIndex"].values
    class_name_list_A = getClasses(config["dataset_A_train_path"]) # sorted
    class_name_list_B = getClasses(config["dataset_B_train_path"]) # sorted
    all_class_name_list = list(set(class_name_list_A+class_name_list_B))
    all_class_name_list.sort()
    # 总分类数
    all_class_nums = len(all_class_name_list)
    generator_A_test = config["generator_A_test"]
    generator_B_test = config["generator_B_test"]
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    # 双方的local to global
    local_to_global_party_A = joblib.load(config["local_to_global_party_A_path"])
    local_to_global_party_B = joblib.load(config["local_to_global_party_B_path"])
    for sample_rate in sample_rate_list:
        for repeat_num in range(10):
            weight_A_path = os.path.join(root_dir, f"{dataset_name}", "HMR", "trained_weights_FangHui", str(int(sample_rate*100)), f"model_A_weight_{repeat_num}.h5")
            weight_B_path = os.path.join(root_dir, f"{dataset_name}", "HMR", "trained_weights_FangHui", str(int(sample_rate*100)), f"model_B_weight_{repeat_num}.h5")
            model_A_extend.load_weights(weight_A_path)
            model_B_extend.load_weights(weight_B_path)
            hmr_eval = HMR_Eval(model_A_extend, model_B_extend, df_test)
            output_predict = hmr_eval.predict_output(
                generator_A_test, 
                generator_B_test, 
                target_size_A, 
                target_size_B, 
                all_class_nums, 
                local_to_global_party_A,
                local_to_global_party_B)
            predict_labels = np.argmax(output_predict, axis = 1)
            # report = classification_report(true_labels, predict_labels, output_dict=True)
            trueOrFalse_list = np.equal(predict_labels,true_labels)
            ans[sample_rate].append(trueOrFalse_list)
    save_dir = os.path.join(root_dir, dataset_name, "HMR")
    save_file_name = f"eval_TrueOrFalse_list_FangHui.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("HMR evaluation FangHui end")
    return ans


def app_HMR_eval_initial():
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    config = animal_2_config
    dataset_name =  config["dataset_name"]
    model_A_extend, model_B_extend = load_models_pool(config) # 构建和编译
    setproctitle.setproctitle(f"{dataset_name}|HMR|eval_init")
    test_dir = f"exp_data/{dataset_name}/sampling/percent/random_split/test"
    df_test = pd.read_csv(os.path.join(test_dir, "test.csv"))
    class_name_list_A = getClasses(config["dataset_A_train_path"]) # sorted
    class_name_list_B = getClasses(config["dataset_B_train_path"]) # sorted
    all_class_name_list = list(set(class_name_list_A+class_name_list_B))
    all_class_name_list.sort()
    # 总分类数
    all_class_nums = len(all_class_name_list)
    generator_A_test = config["generator_A_test"]
    generator_B_test = config["generator_B_test"]
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    # 双方的local to global
    local_to_global_party_A = joblib.load(config["local_to_global_party_A_path"])
    local_to_global_party_B = joblib.load(config["local_to_global_party_B_path"])
    hmr_eval = HMR_Eval(model_A_extend, model_B_extend, df_test)
    acc = hmr_eval.eval(
        generator_A_test, 
        generator_B_test, 
        target_size_A, 
        target_size_B, 
        all_class_nums, 
        local_to_global_party_A,
        local_to_global_party_B)
    print(f"HMR init acc:{acc}")


if __name__ == "__main__":
    # app_HMR_retrain()
    # app_HMR_eval()
    # app_HMR_eval_initial()
    # app_HMR_retrain_FangHui()
    # app_HMR_eval_FangHui()
    app_HMR_eval_TrueFalse_FangHui()
    # app_HMR_eval_classes_FangHui()
