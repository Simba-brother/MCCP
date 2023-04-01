'''
一些辅助函数
'''
import joblib
import Base_acc
import os
import pandas as pd

def convert_data_struct(data, config):
    # 返回结果
    ans = {}
    
    base_acc = config["CFL"]
    base_A_acc = config["A_acc"]
    base_B_acc = config["B_acc"]

    ans["base_acc"] = base_acc
    ans["base_A_acc"] = base_A_acc
    ans["base_B_acc"] = base_B_acc
    ans["retrained_acc"] = {}

    keys = list(data.keys())
    keys.sort()
    for key in keys:
        ans["retrained_acc"][key] = []
        item_list = data[key]
        for item in item_list:
            obj = {}
            obj["improve_acc"] = item["acc_improve"]
            obj["improve_A_acc"] = None
            obj["improve_B_acc"] = None
            ans["retrained_acc"][key].append(obj)
    return ans

def help1():
    # 加载配置
    config = Base_acc.animal_2
    dataset_name = "animal_2"
    # 加载数据
    reTrain_result =  joblib.load(f"exp_data/{dataset_name}/retrainResult/percent/CFL/reTrain_acc_improve_accords.data")
    ans = convert_data_struct(reTrain_result, config)
    save_dir = f"exp_data/{dataset_name}/retrainResult/percent/CFL"
    file_name = "reTrain_acc.data"
    file_path = os.path.join(save_dir,file_name)
    joblib.dump(ans, file_path)
    print("save_success")

   
def help2():
    df = pd.read_csv("/data/mml/overlap_v2_datasets/animal_2/party_B/dataset_split/train_overlap.csv")

    file_path_list = []
    for row_idx, row in df.iterrows():
        old_file_path = row["file_path"]
        new_file_path = old_file_path.replace('data_split' , 'dataset_split')
        file_path_list.append(new_file_path)
    df["file_path"] = file_path_list

    save_dir = "/data/mml/overlap_v2_datasets/animal_2/party_B/dataset_split"
    file_name = "train_overlap.csv"
    file_path = os.path.join(save_dir, file_name)
    df.to_csv(file_path)
    print("help2() finally")

if __name__ == "__main__":
    # help1()
    # help2()
    pass

            

