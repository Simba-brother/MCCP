import tensorflow.keras as keras 
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.optimizers import Adamax
from DataSetConfig import car_body_style_config,flower_2_config,food_config,fruit_config,sport_config,weather_config,animal_config,animal_2_config,animal_3_config


def load_models_pool(config, lr=1e-5):
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

class CutModel(object):
    def __init__(self, model_A, model_B):
        self.model_A = model_A
        self.model_B = model_B
    def cut(self, layer_index_A, layer_index_B):
        model_A_cut = keras.Model(inputs = self.model_A.input, outputs = self.model_A.get_layer(index = layer_index_A).output)  # weather:-2
        model_B_cut = keras.Model(inputs = self.model_B.input, outputs = self.model_B.get_layer(index = layer_index_B).output)
        return model_A_cut, model_B_cut
    
def app_1():
    config = animal_3_config
    dataset_name = config["dataset_name"]
    model_A, model_B = load_models_pool(config)
    cutModel = CutModel(model_A, model_B)
    layer_index_A = config["model_A_cut"]
    layer_index_B = config["model_B_cut"]
    model_A_cut, model_B_cut = cutModel.cut(layer_index_A, layer_index_B)
    root_dir = "/data2/mml/overlap_v2_datasets"
    save_dir = os.path.join(root_dir, dataset_name, "party_A", "cut_model")
    save_file_name = "model_cutted.h5"
    save_file_path = os.path.join(save_dir,save_file_name)
    model_A_cut.save(save_file_path)
    print(f"save_file_path:{save_file_path}")
    save_dir = os.path.join(root_dir, dataset_name, "party_B", "cut_model")
    save_file_name = "model_cutted.h5"
    save_file_path = os.path.join(save_dir,save_file_name)
    model_B_cut.save(save_file_path)
    print(f"save_file_path:{save_file_path}")

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    app_1()
    
