import sys
sys.path.append("./")
import os
import json
import joblib
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import (Model, Sequential, load_model,
                                     model_from_json)
from tensorflow.python.keras.layers.core import Dropout
from yaml import load

from DataSetConfig import (car_body_style_config,
                           flower_2_config,
                           food_config,
                           fruit_config, 
                           sport_config, 
                           weather_config, animal_config, animal_2_config, animal_3_config)




def combination(one_model, two_model, classes_num, model_A_layeIndex, model_B_layerIndex):
    '''
    合并两个模型, 先初始化合并, 不管权重
    '''
    # 保存model_one 权重
    # save_dir = "./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge_model/model_1_info"
    # save_fileName = "one_model_weights.h5"
    # one_model.save_weights(os.path.join(save_dir, save_fileName))
    # # 保存model_one json
    # one_model_json = one_model.to_json()
    # save_fileName = "one_model.json"
    # with open(os.path.join(save_dir, save_fileName), 'w') as file:
    #     file.write(one_model_json)

    # 稍等片刻，修改一下layer_name

    # 读取json结构
    # with open("./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge_model/model_1_info/one_model.json", "r") as f:  # 要变
    #     json_string = f.read()
    # 加载结构
    # one_model = model_from_json(json_string)
    # 加载回权重
    # one_model.load_weights("./saved/custom_split/shape/Geometric_Shapes_Mathematics/train_test/merge_model/model_1_info/one_model_weights.h5")  # 要变
    # simulation既要修改json中的层名，也要遍历model.layers去修改layer._name
    # animal 只要遍历model.layers去修改layer._name
    for layer in one_model.layers:
        layer._name = layer._name + str("A")  # 在simulation,animal, flower场景下好用
    # for w in one_model.weights:
    #     w._handle_name = 'ONE' + w.name
    # for layer in two_model.layers:
    #     layer._name = layer._name + str("TWO")  # 在animal场景下好用
    # for w in two_model.weights:
    #     w._handle_name = 'TWO' + w.name
    # one_model.get_layer(name='lambda_input').name='lambda_input_one'
    
    # 截取掉model_left 的 全连接
    model_left = keras.Model(inputs = one_model.input, outputs = one_model.get_layer(index = model_A_layeIndex).output)  # weather:-2
    # lock
    # for layer in model_left.layers:
    #     layer.trainable = False
    # print(model_left.summary())
  
    
    input_left = model_left.input
    output_left = model_left.output

    # 截取掉model_right 的 全连接
    model_right = keras.Model(inputs = two_model.input, outputs = two_model.get_layer(index = model_B_layerIndex).output)   # weather:-3
    # lock
    # for layer in model_right.layers:
    #     layer.trainable = False
    input_right = model_right.input
    output_right = model_right.output

    # model concat，最后一层隐藏层并起来
    concatenated = keras.layers.concatenate([output_left, output_right])
    # print(concatenated)
    # 搭建网络剩余部分
    # x = Dense(512, activation='relu', name="dense_my_1")(concatenated)
    # x = Dropout(0.02)(x)
    # 连上一个输出层
    final_output = Dense(classes_num, activation='softmax', name="output")(concatenated)  # weather:14
    model = Model(inputs = [input_left, input_right], outputs = final_output)
    print("model_combination() success")
    return model

def modify_combination_weights(model_1, 
                            model_2, 
                            classes_num, 
                            party_A_local_to_global_map, 
                            party_B_local_to_global_map,
                            combination_model
                            ):
    '''
        把权重继承过来
    '''
    # 得到party_A权重数据
    weights_A = model_1.get_layer(index = -1).get_weights()
    weight_array_A =  weights_A[0]
    bias_array_A = weights_A[1]
    # 得到party_B权重数据
    weights_B = model_2.get_layer(index = -1).get_weights()
    weight_array_B =  weights_B[0]
    bias_array_B = weights_B[1]
    # 获得两个模型的最后一层隐藏层神经元数量
    last_hidden_layer_neuron_num_party_A = model_1.get_layer(index = -2).output_shape[1]
    last_hidden_layer_neuron_num_party_B = model_2.get_layer(index = -2).output_shape[1]
    combination_neuron_num = last_hidden_layer_neuron_num_party_A + last_hidden_layer_neuron_num_party_B
    # 获得各方local_labelIndex 其实就是各方的local_neuron_index
    local_labelIndex_party_A_list = sorted(party_A_local_to_global_map.keys(),key=lambda x:x,reverse=False)
    local_labelIndex_party_B_list = sorted(party_B_local_to_global_map.keys(),key=lambda x:x,reverse=False)
    # 拼接出的新权重数据结构
    new_weights = [None, None]
    new_weights_array = np.zeros((combination_neuron_num, classes_num),dtype=np.float32)
    new_bias_array = np.zeros(classes_num, dtype=np.float32)
    # 继承party_A的输出层神经元权重
    for local_labelIndex in local_labelIndex_party_A_list:
        local_neuron_index = local_labelIndex
        global_neuron_index = party_A_local_to_global_map[local_neuron_index]
        # 获得这个输出神经元的 权重连接
        cur_neuron_weights = weight_array_A[:,local_neuron_index]
        cur_neuron_bias = bias_array_A[local_neuron_index]
        # 去填充 新的权重矩阵
        new_weights_array[0:last_hidden_layer_neuron_num_party_A, global_neuron_index] = cur_neuron_weights
        new_bias_array[global_neuron_index] = cur_neuron_bias
    # 继承party_B的输出层神经元权重
    for local_labelIndex in local_labelIndex_party_B_list:
        local_neuron_index = local_labelIndex
        global_neuron_index = party_B_local_to_global_map[local_neuron_index]
    
        # 获得这个输出神经元的 权重连接
        cur_neuron_weights = weight_array_B[:,local_neuron_index]
        cur_neuron_bias = bias_array_B[local_neuron_index]
        # 去填充 新的权重矩阵
        new_weights_array[last_hidden_layer_neuron_num_party_A:, global_neuron_index] = cur_neuron_weights
        new_bias_array[global_neuron_index] = cur_neuron_bias

    new_weights[0] = new_weights_array
    new_weights[1] = new_bias_array
    # 设置新的权重
    combination_model.get_layer(index = -1).set_weights(new_weights)
    # 冻结前面的层
    for layer in combination_model.layers[:-1]:
        layer.trainable = False
    print("modify_combination_weights() successs")
    return combination_model

def modifyModel(model, layer_index, classes_num, local_to_global_map):
    '''
    扩展单方模型的分类空间到总空间
    '''
    # 获得原来模型输出层权重数据
    weights = model.get_layer(index = -1).get_weights()
    weight_array =  weights[0]
    bias_array = weights[1]
    # 获得最后一层隐藏层神经元数量
    last_hidden_layer_neuron_num = model.get_layer(index = -2).output_shape[1]
    # 获得local_label_index_list
    local_labelIndex_list = sorted(local_to_global_map.keys(),key=lambda x:x,reverse=False)
    # 拼接出的新权重数据结构
    new_weights = [None, None]
    new_weights_array = np.zeros((last_hidden_layer_neuron_num, classes_num),dtype=np.float32)
    new_bias_array = np.zeros(classes_num, dtype=np.float32)
    # 继承的输出层神经元权重(注意新分类的字典序列)
    for local_labelIndex in local_labelIndex_list:
        local_neuron_index = local_labelIndex
        global_neuron_index = local_to_global_map[local_neuron_index]
        # 获得这个输出神经元的 权重连接
        cur_neuron_weights = weight_array[:,local_neuron_index]
        cur_neuron_bias = bias_array[local_neuron_index]
        # 去填充 新的权重矩阵
        new_weights_array[:, global_neuron_index] = cur_neuron_weights
        new_bias_array[global_neuron_index] = cur_neuron_bias

    new_weights[0] = new_weights_array
    new_weights[1] = new_bias_array
    # cut 掉 output 链接上新的output
    model_cut = keras.Model(inputs = model.input, outputs = model.get_layer(index = layer_index).output)
    # model_cut.add(Dense(classes_num, activation='softmax', name="dense")) # -2
    final_output = Dense(classes_num, activation='softmax', name="dense_new")(model_cut.output)
    model_final = Model(inputs = model_cut.input, outputs = final_output)
    # 设置新的权重
    model_final.get_layer(index = -1).set_weights(new_weights)
    # 冻结前面的层
    for layer in model_final.layers[:-1]:
        layer.trainable = False

    print("modifyModel success")
    return model_final


def load_teacher_models(config, lr=1e-5):
    # 加载模型
    model_A = load_model(config["model_A_struct_path"])
    if not config["model_A_weight_path"] is None:
        model_A.load_weights(config["model_A_weight_path"])

    model_B = load_model(config["model_B_struct_path"])
    if not config["model_B_weight_path"] is None:
        model_B.load_weights(config["model_B_weight_path"])
    # model_A = add_reserve_class(model_A)
    # model_B = add_reserve_class(model_B)
    # model_A.compile(loss=categorical_crossentropy,
    #             optimizer=Adamax(learning_rate=lr),
    #             metrics=['accuracy'])
    # model_B.compile(loss=categorical_crossentropy,
    #             optimizer=Adamax(learning_rate=lr),
    #             metrics=['accuracy'])
    # # 各方模型池
    # models_pool = [model_A, model_B]
    # # 模型池中的模型是否具备保留类功能
    # hasReceived_pool = [True, True]
    return model_A,model_B


def app_join_teacher_models(config, root_dir):
    '''
    生成MCCP级联model
    '''
    # 加载 双方模型
    model_A, model_B = load_teacher_models(config)
    # 简单合并模型
    merged_classes_num = 9  # 100分类+22分类-13重叠分类
    model_A_layerIndex = config["model_A_cut"]
    model_B_layerIndex = config["model_B_cut"]
    combined_model_randomWeights = combination(model_A, model_B, merged_classes_num, model_A_layerIndex, model_B_layerIndex)
    # save_dir = "/data/mml/overlap_v2_datasets/animal_3/merged_model"
    # combined_model_randomWeights.save(os.path.join(save_dir,"combination_Model_randomWeights.h5"))

    local_to_gobal_party_A = joblib.load(config["local_to_global_party_A_path"])
    local_to_gobal_party_B = joblib.load(config["local_to_global_party_B_path"])

    # 得到权重继承模型
    combined_model_inheritWeights = modify_combination_weights(
                            model_A,
                            model_B, 
                            merged_classes_num, 
                            local_to_gobal_party_A, 
                            local_to_gobal_party_B,
                            combined_model_randomWeights
                        )
    # combined_model_inheritWeights.save(os.path.join(save_dir,"combination_model_inheritWeights.h5"))
    return combined_model_randomWeights, combined_model_inheritWeights

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    config = car_body_style_config
    root_dir = "/data2/mml/overlap_v2_datasets"
    '''
    MCCP模型合并
    '''
    # app_join_teacher_models(config, root_dir)
    '''
    单方模型扩展
    '''
    # model = load_model("/data/mml/overlap_v2_datasets/weather/party_B/models/model_struct/ResNet152V2.h5")
    # model.load_weights("/data/mml/overlap_v2_datasets/weather/party_B/models/model_weights/model_weight_049_0.8969.h5")
    # layer_index = -2 # 只 cut output层
    # classes_num = 13
    # local_to_global_map = joblib.load("exp_data/weather/LocalToGlobal/local_to_gobal_party_B.data")
    # model_extended = modifyModel(model, layer_index, classes_num, local_to_global_map)
    # save_dir = "/data/mml/overlap_v2_datasets/weather/merged_model"
    # file_name = "model_B_extended.h5"
    # model_extended.save(os.path.join(save_dir, file_name))

    
   