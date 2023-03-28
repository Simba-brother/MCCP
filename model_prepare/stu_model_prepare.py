from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
def add_last_layer(model, class_num):
    
    mml_classifier = Dense(class_num, activation='softmax', name="mml_classifier")(model.output)
    new_model = Model(inputs = model.input, outputs = mml_classifier)
    return new_model

if __name__ == "__main__":
    # 加载预训练noTop model
    pre_model = load_model("/data/mml/some_models/DenseNet121_224_224_noTop.h5")
    # 添加classifer layer
    class_num = 16
    my_model = add_last_layer(pre_model, class_num)
    save_dir = "/data/mml/overlap_v2_datasets/sport/merged_model"
    file_name = f"DenseNet121_224_224_claNum_{class_num}.h5"
    save_path = os.path.join(save_dir, file_name)
    my_model.save(save_path)
    print("success")
