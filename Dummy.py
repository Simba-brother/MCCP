import os
import setproctitle
import numpy as np
import joblib
from DataSetConfig import car_body_style_config,flower_2_config,food_config
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adamax


def load_models(config, lr=1e-3):
    # 加载模型
    model_A = load_model(config["model_A_struct_path"])
    if not config["model_A_weight_path"] is None:
        model_A.load_weights(config["model_A_weight_path"])

    model_B = load_model(config["model_B_struct_path"])
    if not config["model_B_weight_path"] is None:
        model_B.load_weights(config["model_B_weight_path"])

    model_A.compile(loss=categorical_crossentropy,
                optimizer=Adamax(learning_rate=lr),
                metrics=['accuracy'])
    model_B.compile(loss=categorical_crossentropy,
                optimizer=Adamax(learning_rate=lr),
                metrics=['accuracy'])
    return model_A,model_B

def get_confidences(model, df, generator, target_size):
   
   batch_size = 5
   batches = generator.flow_from_dataframe(df, 
                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                            x_col='file_path', y_col="label", 
                            target_size=target_size, class_mode='categorical', # one-hot
                            color_mode='rgb', classes=None,
                            shuffle=False, batch_size=batch_size,
                            validate_filenames=False)
   probs = model.predict_generator(generator = batches, steps=batches.n/batch_size)
   confidences = np.max(probs, axis = 1)
   pseudo_label_indexes = np.argmax(probs, axis=1)
   return confidences, pseudo_label_indexes

class Dummy(object):
    def __init__(self, model_A, model_B, df_test):
        self.model_A = model_A
        self.model_B = model_B
        self.df_test = df_test
    def integrate(self, generator_A_test,  generator_B_test, target_size_A, target_size_B, local_to_global_A, local_to_global_B):
        pseudo_labels = []
        confidences_A, pseudo_labels_A = get_confidences(self.model_A, self.df_test, generator_A_test, target_size_A)
        confidences_B, pseudo_labels_B = get_confidences(self.model_B, self.df_test, generator_B_test, target_size_B)
        for i in range(confidences_A.shape[0]):
            if confidences_A[i] > confidences_B[i]:
                pseudo_global_label = local_to_global_A[pseudo_labels_A[i]]
                pseudo_labels.append(pseudo_global_label)
                
            else:
                pseudo_global_label = local_to_global_B[pseudo_labels_B[i]]
                pseudo_labels.append(pseudo_global_label)
        ground_truths = self.df_test["label_globalIndex"].to_numpy(dtype="int")
        pseudo_labels = np.array(pseudo_labels)
        acc = np.sum(pseudo_labels == ground_truths) / self.df_test.shape[0]
        acc = round(acc,4)
        return acc
    
def app_Dummy():
    
    config = car_body_style_config
    dataset_name = config["dataset_name"]
    setproctitle.setproctitle("{dataset_name}|Dummy")
    test_dir = f"exp_data/{dataset_name}/sampling/percent/random_split/test"
    df_test = pd.read_csv(os.path.join(test_dir, "test.csv"))
    generator_A_test = config["generator_A_test"]
    generator_B_test = config["generator_B_test"]
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    local_to_global_A = joblib.load(config["local_to_global_party_A_path"])
    local_to_global_B = joblib.load(config["local_to_global_party_B_path"])
    model_A, model_B = load_models(config=config)
    dummy = Dummy(model_A, model_B, df_test)
    acc = dummy.integrate(generator_A_test,  generator_B_test, target_size_A, target_size_B, local_to_global_A, local_to_global_B)
    return acc
if __name__ == "__main__":
    # app_Dummy()
    pass
