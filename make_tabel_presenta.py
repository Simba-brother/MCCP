import pandas as pd
import numpy as np
import joblib
import os
# from tabulate import tabulate
# print(tabulate(df,tablefmt="latex_raw"))
def get_y_list(dic):
    ans = []
    base_acc = dic["base_acc"]
    retrain_dic = dic["retrained_acc"]
    keys = list(retrain_dic.keys())
    keys.sort()
    for key in keys:
        sum = 0
        item_list = retrain_dic[key]
        for item in item_list:
            improve_acc = item["improve_acc"]
            acc = base_acc + improve_acc
            sum += acc
        avg = round( sum / len(item_list), 4)
        ans.append(avg)
    return ans

df = pd.DataFrame(np.random.randint(80, 120, size=(6, 6)), 
                   columns= ["\\textbf{30}",60,90,120,150,180],
                   index=pd.MultiIndex.from_product([["car","flower_2"],
                                                    ["OurCombin","HMR", "CFL"]]))




def make_tabel_1():
    # 构建出表格结构
    df = pd.DataFrame(np.random.randint(2, 3, size=(27, 9)), 
        columns= ["1%", "3%", "5%", "10%", "15%", "20%", "50%", "80%", "100%"],
        index=pd.MultiIndex.from_product([["car_body_style","flower_2","food", "Fruit", "sport", "weather", "animal", "animal_2", "animal_3"],
                                        ["HMR", "CFL","OurCombin"]]))
    # 加载数据
    for dataset_name in dataset_name_list:    
        cfl_data = joblib.load(f"exp_data/{dataset_name}/retrainResult/{sample_method}/CFL/reTrain_acc.data")
        hmr_data = joblib.load(f"exp_data/{dataset_name}/retrainResult/{sample_method}/HMR/reTrain_acc.data")
        ourCombin_data = joblib.load(f"exp_data/{dataset_name}/retrainResult/{sample_method}/OurCombin/reTrain_acc.data")
        cfl_list = get_y_list(cfl_data)
        hmr_list = get_y_list(hmr_data)
        ourCombin_list = get_y_list(ourCombin_data)

        df.loc[dataset_name,"HMR"] = hmr_list
        df.loc[dataset_name,"CFL"] = cfl_list
        df.loc[dataset_name,"OurCombin"] = ourCombin_list
    return df

# 全局变量
dataset_name_list = ["car_body_style", "flower_2", "food", "Fruit", "sport", "weather", "animal", "animal_2", "animal_3"]
sample_method = "percent"


if __name__ == "__main__":
    # df = make_tabel_1()
    # save_dir = "exp_tabel"
    # file_name = "tabel_1.xls"
    # file_path = os.path.join(save_dir, file_name)
    # df.to_excel(file_path,sheet_name="Sheet2")
    # print("success")
    pass