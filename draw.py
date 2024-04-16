'''
画图文件
'''
import joblib
from collections import defaultdict
from matplotlib import pyplot as plt
import os
import numpy as np
import seaborn as sns
import pandas as pd
from DataSetConfig import food_config, fruit_config, sport_config, weather_config, flower_2_config, car_body_style_config, animal_config, animal_2_config, animal_3_config


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
        
def draw_lines(x_list, y_data):
    ourCombin_list = y_data["OurCombin"]
    LR_list = y_data["LR"]
    DT_list = y_data["DT"]
    # 画布
    fig = plt.figure(figsize=(3,3), dpi=600)
    # 激活区域
    # plt.subplot(1,2,1)
    # plt.xticks(x_list)
    # 画线
    line_1 = plt.plot(x_list, ourCombin_list, label = "MCCP", color = "red", marker = "x")
    line_2 = plt.plot(x_list, LR_list, label = "LogisticRegression", color = "green", marker = "o")
    line_3 = plt.plot(x_list, DT_list, label = "DecisionTree", color = "blue", marker = "s")

    # 画网格
    plt.grid()
    # 画水平线
    # plt.axhline(y_data["OurCombin_base_acc"],color='lightcoral',ls='--')
    # plt.axhline(y_data["HMR_base_acc"],color='springgreen',ls='--')
    # plt.axhline(y_data["Dummy_base_acc"], color = "orange", ls="solid", marker="^", label="Dummy")
    # 图例
    plt.legend()
    # 坐标轴说明
    plt.xlabel("sampling rate", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    # plt.tick_params(labelsize=15)  #刻度字体大小13
    plt.xticks(fontproperties = "Times New Roman", fontsize = 11)
    # 图题目
    # plt.title("Accuracy of Knowledge Amalgamation")
    return fig

def draw_truncation_line(x_list, y_data):
    ourCombin_list = y_data["OurCombin"]
    hmr_list = y_data["HMR"]
    cfl_list = y_data["CFL"]
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(3,3), dpi=600)  # 绘制两个子图
    plt.xlabel("Sampling ratio", fontproperties = "Times New Roman", fontsize = 12)
    plt.ylabel("Accuracy", fontproperties = "Times New Roman", fontsize = 12)
    plt.xticks(fontproperties = "Times New Roman", fontsize = 11)
    plt.xticks(fontproperties = "Times New Roman", fontsize = 11)

    ax1.xaxis.tick_top()
    ax1.xaxis.set_visible(True)
    ax2.xaxis.set_visible(True)
    plt.subplots_adjust(wspace=0,hspace=0.08) # 设置子图间距

    ax2.plot(x_list, cfl_list, label = "CFL", color = "blue", marker = 's')

    ax1.plot(x_list, ourCombin_list, label = "MCCP", color = "red", marker = "x")   # 绘制折线
    ax1.plot(x_list, hmr_list, label = "HMR", color = "green", marker = "o")
    ax1.plot(x_list, cfl_list, label = "CFL", color = "blue", marker = 's')
    ax1.axhline(y_data["Dummy_base_acc"], color = "orange", ls="solid", marker="^", label="Dummy")
    ax1.set_ylim(0.80,0.90) # 设置纵坐标范围
    # car:(0.75,0.85)
    # flower:(0.85,0.92) rebuttal：(0.84,0.92)
    # food:(0.80,0.95) rebuttal:(0.7,0.95)
    # fruit:(0.80,1.0)
    # sport:(0.7,0.90)
    # weather:(0.7,0.82) rebuttal:(0.7,0.85)
    # animal_1:(0.81,0.83) rebuttal:(0.81,0.84)
    # animal_2:(0.80,0.90)
    # animal_3:(0.70,0.86)
    # 画网格
    ax1.grid(axis="both")
    ax2.grid(axis="both")

    ax1.legend() # 让图例生效


    ax2.spines['top'].set_visible(False)    # 边框控制
    ax2.spines['bottom'].set_visible(True) # 边框控制
    ax2.spines['right'].set_visible(True)  # 边框控制

    ax1.spines['top'].set_visible(True)   # 边框控制
    ax1.spines['bottom'].set_visible(False) # 边框控制
    ax1.spines['right'].set_visible(True)  # 边框控制
    # ax2.tick_params(labeltop='off') 
    # 绘制断层线
    d = 0.01  # 断层线的大小
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs) 
    kwargs.update(transform=ax2.transAxes, color='k')  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs) 
    return f

def draw_line_main_2(config):
    def get_y_list_internal(data):
        y_list = []
        sample_rate_list = [0.01,0.03,0.05,0.1,0.15,0.2]
        for sample_rate in sample_rate_list:
            avg = sum(data[sample_rate])/len(data[sample_rate])
            y_list.append(avg)
        return y_list
    
    dataset_name = config["dataset_name"]
    x_list = ["1%","3%","5%", "10%", "15%", "20%"]
    root_dir = "/data2/mml/overlap_v2_datasets"
    MCCP_res = joblib.load(os.path.join(root_dir, dataset_name, "MCCP", "eval_ans_FangHui.data"))
    HMR_res = joblib.load(os.path.join(root_dir, dataset_name, "HMR", "eval_ans_FangHu.data"))
    CFL_res = joblib.load(os.path.join(root_dir, dataset_name, "CFL", "eval_ans_FangHui.data"))
    Dummy_res = joblib.load(os.path.join(root_dir, dataset_name, "Dummy", "eval_ans_FangHui.data"))
    MCCP_y_list = get_y_list_internal(MCCP_res)
    HMR_y_list = get_y_list_internal(HMR_res)
    CFL_y_list = get_y_list_internal(CFL_res)
    y_data = {}
    y_data["OurCombin"] = MCCP_y_list
    y_data["HMR"] = HMR_y_list
    y_data["CFL"] = CFL_y_list
    y_data["Dummy_base_acc"] = Dummy_res
    fig = draw_truncation_line(x_list, y_data)
    # 保存图片
    save_dir = f"exp_image/{dataset_name}"
    file_name = f"RQ2_rebuttal_FangHui_repeat10.pdf"
    file_path = os.path.join(save_dir, file_name)
    fig.savefig(file_path,bbox_inches="tight",pad_inches=0.1)

def draw_line_main_ML(config):
    def get_y_list_internal(data):
        y_list = []
        sample_rate_list = [0.01,0.03,0.05,0.1,0.15,0.2]
        for sample_rate in sample_rate_list:
            avg = sum(data[sample_rate])/len(data[sample_rate])
            y_list.append(avg)
        return y_list
    
    dataset_name = config["dataset_name"]
    x_list = ["1%","3%","5%", "10%", "15%", "20%"]
    root_dir = "/data2/mml/overlap_v2_datasets"
    MCCP_res = joblib.load(os.path.join(root_dir, dataset_name, "MCCP", "eval_ans_FangHui.data"))
    LogisticRegression_res = joblib.load(os.path.join(root_dir, dataset_name, "LogisticRegression", "eval_ans.data"))
    DecisionTree_res = joblib.load(os.path.join(root_dir, dataset_name, "DecisionTree", "eval_ans.data"))
    MCCP_y_list = get_y_list_internal(MCCP_res)
    LR_y_list = get_y_list_internal(LogisticRegression_res)
    DT_y_list = get_y_list_internal(DecisionTree_res)
    y_data = {}
    y_data["OurCombin"] = MCCP_y_list
    y_data["LR"] = LR_y_list
    y_data["DT"] = DT_y_list
    fig = draw_lines(x_list, y_data)
    # 保存图片
    save_dir = f"exp_image/{dataset_name}"
    file_name = f"RQ2_MCCP_LR_DT.pdf"
    file_path = os.path.join(save_dir, file_name)
    fig.savefig(file_path,bbox_inches="tight",pad_inches=0.1)

def draw_line_main(config):
    dataset_name = config["dataset_name"]
    sample_method = "percent"
    resulst_file_name = "reTrain_acc.data"
    # x轴(x_list)
    x_list = ["1%","3%","5%", "10%", "15%", "20%"] # , "50%", "80", "100%"
    # x_list = [30,60,90,120,150,180]
    # 加载数据
    OurCombin = joblib.load(f"exp_data/{dataset_name}/retrainResult/{sample_method}/OurCombin/{resulst_file_name}")
    HMR = joblib.load(f"exp_data/{dataset_name}/retrainResult/{sample_method}/HMR/{resulst_file_name}")
    CFL = joblib.load(f"exp_data/{dataset_name}/retrainResult/{sample_method}/CFL/{resulst_file_name}")
    Dummy = joblib.load(f"exp_data/{dataset_name}/retrainResult/dummy/dummy.data")
    # 获得y_list
    ourCombin_list  = get_y_list(OurCombin)
    hmr_list = get_y_list(HMR)
    cfl_list = get_y_list(CFL)
    # 封装字典
    y_data = {}
    y_data["OurCombin"] = ourCombin_list[:-3]
    y_data["HMR"] = hmr_list[:-3]
    y_data["CFL"] = cfl_list[:-3]
    y_data["OurCombin_base_acc"] = OurCombin["base_acc"]
    y_data["HMR_base_acc"] = HMR["base_acc"]
    y_data["CFL_base_acc"] = CFL["base_acc"]
    y_data["Dummy_base_acc"] = Dummy["combin_acc"]
    # 画图
    fig = draw_truncation_line(x_list, y_data)
    # 保存图片
    save_dir = f"exp_image/{dataset_name}"
    file_name = f"RQ2_new.pdf"
    file_path = os.path.join(save_dir, file_name)
    fig.savefig(file_path,bbox_inches="tight",pad_inches=0.1)

def draw_case_study(config):
    dataset_name = config["dataset_name"]
    sample_method = "percent"
    resulst_file_name = "reTrain_acc.data"
    # x轴(x_list)
    x_list = ["1%","3%","5%", "10%", "15%", "20%"] # "50%", "80", "100%"
    color = [
    '#FF0000',
    '#00ff00',
    '#0000ff']
    # x_list = [30,60,90,120,150,180]
    # 加载数据
    OurCombin = joblib.load(f"exp_data/{dataset_name}/retrainResult/{sample_method}/OurCombin/{resulst_file_name}")
    HMR = joblib.load(f"exp_data/{dataset_name}/retrainResult/{sample_method}/HMR/{resulst_file_name}")
    CFL = joblib.load(f"exp_data/{dataset_name}/retrainResult/{sample_method}/CFL/{resulst_file_name}")
    Dummy = joblib.load(f"exp_data/{dataset_name}/retrainResult/dummy/dummy.data")
    # 获得y_list
    ourCombin_list  = get_y_list(OurCombin)
    ourCombin_list = ourCombin_list[:-3]
    hmr_list = get_y_list(HMR)
    hmr_list = hmr_list[:-3]
    cfl_list = get_y_list(CFL)
    cfl_list = cfl_list[:-3]
    dummy_list = [Dummy["combin_acc"]]*len(x_list)
    our_hmr_ans = []
    our_cfl_ans = []
    our_dummy_ans = []
    for i in range(len(ourCombin_list)):
        dif_1 = ourCombin_list[i] - hmr_list[i]
        dif_2 = ourCombin_list[i] - cfl_list[i]
        dif_3 = ourCombin_list[i] - dummy_list[i]
        dif_1 = round(dif_1,4)
        dif_2 = round(dif_2,4)
        dif_3 = round(dif_3,4)
        our_hmr_ans.append(dif_1)
        our_cfl_ans.append(dif_2)
        our_dummy_ans.append(dif_3)
    # fig = plt.figure(figsize=(7,5))  
    fig,ax1 = plt.subplots()
    ax1.set_ylabel('Improved accuracy')
    ax1.set_xlabel('percent of sampling')
    ax1.set_ylim([0.02,0.1])
    x_1 = list(range(len(x_list)))
    bar_width = 0.2 # 柱子宽度
    ax1.bar(x_1, our_hmr_ans, width=bar_width, label="MCCP-HMR", color=color[0])
    # 第二个柱子的位置
    x_2 = list(range(len(x_list)))
    for i in range(len(x_2)):
        x_2[i] = x_1[i]+bar_width
    plt.bar(x_2, our_dummy_ans, width=bar_width, label="MCCP-Dummy", tick_label = x_list, color=color[1])
    # 第三个柱子的位置
    ax2 = ax1.twinx()
    ax2.set_ylabel('Improved accuracy')
    ax2.set_ylim([0.2,0.6])
    # ax2.set_ylim([0.02,0.1])
    x_3 = list(range(len(x_list)))
    for i in range(len(x_3)):
        x_3[i] = x_1[i]+2*bar_width
    ax2.bar(x_3, our_cfl_ans, width=bar_width, label="MCCP-CFL", color=color[2])
    # plt.xticks(rotation=-15)  
    # plt.ylim(0.02, 0.1)  
    fig.tight_layout()
    # plt.legend()
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.show()
    save_dir = f"exp_image/animal_2"
    file_name = "case_study.pdf"
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)
    
def draw_box():
    dataset_name = config["dataset_name"]
    labels = '1%', '3%', '5%', '10%', '15%', '20%'
    train_acc = joblib.load(f"exp_data/{dataset_name}/retrainResult/percent/OurCombin/train_acc_v3.data")
    A = train_acc["train"][1]
    # 计算方差
    var_A = np.var(A)
    B = train_acc["train"][3]
    var_B = np.var(B)
    C = train_acc["train"][5]
    var_C = np.var(C)
    D = train_acc["train"][10]
    var_D = np.var(D)
    E = train_acc["train"][15]
    var_E = np.var(E)
    F = train_acc["train"][20]
    var_F = np.var(F)
    print(format(var_A,".3E"))
    print(format(var_B,".3E"))
    print(format(var_C,".3E"))
    print(format(var_D,".3E"))
    print(format(var_E,".3E"))
    print(format(var_F,".3E"))
    data = [A, B, C, D, E, F]
    plt.figure(figsize=(3,3),dpi=600)
    plt.grid(True)  # 显示网格
    plt.boxplot(data,
                widths=0.4,
                medianprops={'color': 'red', 'linewidth': '1.5'}, # 设置中位数的属性，如线的类型、粗细等；
                showmeans=True,
                meanline=False,
                # meanline=True,
                # meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10}, # 设置异常值的属性，如异常点的形状、大小、填充色等
                labels=labels)
    # plt.yticks(np.arange(0.4, 0.81, 0.1))
    plt.xticks(fontproperties = 'Times New Roman', size = 11)
    plt.yticks(fontproperties = 'Times New Roman', size = 11)
    plt.xlabel("Sampling ratio", fontproperties = "Times New Roman", fontsize = 12)
    plt.ylabel("Accuracy", fontproperties = "Times New Roman", fontsize = 12)
    plt.show()
    save_dir = f"exp_image/{dataset_name}"
    file_name = "box.pdf"
    file_path = os.path.join(save_dir, file_name)
    # plt.savefig(file_path, bbox_inches = 'tight',pad_inches = 0.1)
    print("draw_box successfully!")

def draw_overlap_unqiue_initAcc_bar():
    dataset_name_list = ["car_body_style", "flower_2", "food", "Fruit", "sport", "weather", "animal", "animal_2", "animal_3"]
    alias_list = ["Car", "Flower", "Food", "Fruit", "Sport", "Weather", "Animal_1", "Animal_2", "Animal_3"]
    overlap_list = []
    unique_list = []
    for dataset_name in dataset_name_list:
        initAcc = joblib.load(f"exp_data/{dataset_name}/initAcc.data") 
        overlap_list.append(initAcc["overlap_initAcc"]["accuracy"])
        unique_list.append(initAcc["unique_initAcc"]["accuracy"])
    fig = plt.figure(figsize=(7,5))  
    x = list(range(len(dataset_name_list)))
    bar_width = 0.2 # 柱子宽度
    plt.bar(x, unique_list, width=bar_width, label="unique", tick_label = alias_list, fc="red")
    # 第二个柱子的位置
    x_1 = list(range(len(dataset_name_list)))
    for i in range(len(x_1)):
        x_1[i] = x[i]+bar_width
    plt.bar(x_1, overlap_list, width=bar_width, label="overlapping", fc="green")
    plt.xticks(rotation=-15)  
    plt.ylim(0.4, 1.0)  
    plt.legend()
    plt.show()
    save_dir = f"exp_image/all"
    file_name = "overlap_unique_initAcc.pdf"
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)
    print("draw_overlap_unqiue_initAcc_bar() successfully!")

def draw_overlap_initAcc_bar():
    dataset_name_list = ["car_body_style", "flower_2", "food", "Fruit", "sport", "weather", "animal", "animal_2", "animal_3"]
    alias_list = ["Car", "Flower", "Food", "Fruit", "Sport", "Weather", "Animal_1", "Animal_2", "Animal_3"]
    model_A_list = []
    model_B_list = []
    mccp_list = []
    root_dir = "/data2/mml/overlap_v2_datasets"
    for dataset_name in dataset_name_list:
        overlap_init_acc_teachers =  joblib.load(os.path.join(root_dir,dataset_name,"OriginModel","eval_overlap_merged_test.data"))
        acc_t1 = overlap_init_acc_teachers["acc_A"]
        acc_t2 = overlap_init_acc_teachers["acc_B"]
        acc_mccp = joblib.load(os.path.join(root_dir, dataset_name, "MCCP", "init_overlap_merged_test_acc.data"))
        model_A_list.append(acc_t1)
        model_B_list.append(acc_t2)
        mccp_list.append(acc_mccp)
    fig = plt.figure(figsize=(10,5))  
    x_1 = list(range(len(dataset_name_list)))
    bar_width = 0.2 # 柱子宽度
    plt.bar(x_1, model_A_list, width=bar_width, label="Model A", fc="red")
    # 第二个柱子的位置
    x_2 = list(range(len(dataset_name_list)))
    for i in range(len(x_2)):
        x_2[i] = x_2[i]+bar_width
    plt.bar(x_2, model_B_list, width=bar_width, label="Model B",tick_label = alias_list, fc="green")
    # 第三个柱子的位置
    x_3 = list(range(len(dataset_name_list)))
    for i in range(len(x_3)):
        x_3[i] = x_3[i]+2*bar_width
    plt.bar(x_3, mccp_list, width=bar_width, label="MCCP", fc="blue")
    plt.xticks(rotation=-15)  
    # plt.ylim(0.4, 1.0)  
    plt.legend()
    plt.tight_layout()
    plt.show()
    save_dir = f"exp_image/all"
    file_name = "overlap_initAcc.pdf"
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)
    print("draw_overlap_initAcc_bar() successfully!")

def draw_unique_initAcc_bar():
    dataset_name_list = ["car_body_style", "flower_2", "food", "Fruit", "sport", "weather", "animal", "animal_2", "animal_3"]
    alias_list = ["Car", "Flower", "Food", "Fruit", "Sport", "Weather", "Animal_1", "Animal_2", "Animal_3"]
    model_A_list = []
    model_B_list = []
    mccp_list = []
    root_dir = "/data2/mml/overlap_v2_datasets"
    for dataset_name in dataset_name_list:
        overlap_init_acc_teachers =  joblib.load(os.path.join(root_dir,dataset_name,"OriginModel","eval_unique_merged_test.data"))
        acc_t1 = overlap_init_acc_teachers["acc_A"]
        acc_t2 = overlap_init_acc_teachers["acc_B"]
        acc_mccp = joblib.load(os.path.join(root_dir, dataset_name, "MCCP", "init_unique_merged_test_acc.data"))
        model_A_list.append(acc_t1)
        model_B_list.append(acc_t2)
        mccp_list.append(acc_mccp)
    fig = plt.figure(figsize=(10,5))  
    x_1 = list(range(len(dataset_name_list)))
    bar_width = 0.2 # 柱子宽度
    plt.bar(x_1, model_A_list, width=bar_width, label="Model A", fc="red")
    # 第二个柱子的位置
    x_2 = list(range(len(dataset_name_list)))
    for i in range(len(x_2)):
        x_2[i] = x_2[i]+bar_width
    plt.bar(x_2, model_B_list, width=bar_width, label="Model B",tick_label = alias_list, fc="green")
    # 第三个柱子的位置
    x_3 = list(range(len(dataset_name_list)))
    for i in range(len(x_3)):
        x_3[i] = x_3[i]+2*bar_width
    plt.bar(x_3, mccp_list, width=bar_width, label="MCCP", fc="blue")
    plt.xticks(rotation=-15)  
    # plt.ylim(0.4, 1.0)  
    plt.legend()
    plt.tight_layout()
    plt.show()
    save_dir = f"exp_image/all"
    file_name = "unique_initAcc.pdf"
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)
    print("draw_unique_initAcc_bar() successfully!")

def draw_merged_initAcc_bar():
    dataset_name_list = ["car_body_style", "flower_2", "food", "Fruit", "sport", "weather", "animal", "animal_2", "animal_3"]
    alias_list = ["Car", "Flower", "Food", "Fruit", "Sport", "Weather", "Animal_1", "Animal_2", "Animal_3"]
    model_A_list = []
    model_B_list = []
    mccp_list = []
    root_dir = "/data2/mml/overlap_v2_datasets"
    for dataset_name in dataset_name_list:
        overlap_init_acc_teachers =  joblib.load(os.path.join(root_dir,dataset_name,"OriginModel","eval_merged_test.data"))
        acc_t1 = overlap_init_acc_teachers["acc_A"]
        acc_t2 = overlap_init_acc_teachers["acc_B"]
        acc_mccp = joblib.load(os.path.join(root_dir, dataset_name, "MCCP", "init_merged_test_acc.data"))
        model_A_list.append(acc_t1)
        model_B_list.append(acc_t2)
        mccp_list.append(acc_mccp)
    fig = plt.figure(figsize=(10,5))  
    x_1 = list(range(len(dataset_name_list)))
    bar_width = 0.2 # 柱子宽度
    plt.bar(x_1, model_A_list, width=bar_width, label="Model A", fc="red")
    # 第二个柱子的位置
    x_2 = list(range(len(dataset_name_list)))
    for i in range(len(x_2)):
        x_2[i] = x_2[i]+bar_width
    plt.bar(x_2, model_B_list, width=bar_width, label="Model B",tick_label = alias_list, fc="green")
    # 第三个柱子的位置
    x_3 = list(range(len(dataset_name_list)))
    for i in range(len(x_3)):
        x_3[i] = x_3[i]+2*bar_width
    plt.bar(x_3, mccp_list, width=bar_width, label="MCCP", fc="blue")
    plt.xticks(rotation=-15)  
    # plt.ylim(0.4, 1.0)  
    plt.legend()
    plt.tight_layout()
    plt.show()
    save_dir = f"exp_image/all"
    file_name = "merged_initAcc.pdf"
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)
    print("draw_merged_initAcc_bar() successfully!")

def draw_case_study_bar():
    def get_class_avg_precision(report):
        ans = []
        reports = report[rate]
        for class_i in range(global_class_num):
            precision_list = []
            for report in reports:
                precision = report[str(class_i)]["recall"]
                precision_list.append(precision)
            precision_list = precision_list[0:5]
            avg_precision = sum(precision_list)/len(precision_list)
            ans.append(avg_precision)           
        return ans

    # class_list = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4", "Class_5","Class_6","Class_7","Class_8"]
    class_list = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4", "Class_5"]

    # alias_list = ["Car", "Flower", "Food", "Fruit", "Sport", "Weather", "Animal_1", "Animal_2", "Animal_3"]
    root_dir = "/data2/mml/overlap_v2_datasets"
    dataset_name = "animal_2"
    rate = 0.03
    global_class_num = 6
    MCCP_report =  joblib.load(os.path.join(root_dir,dataset_name,"MCCP","eval_classes_FangHui.data"))
    HMR_report =  joblib.load(os.path.join(root_dir,dataset_name,"HMR","eval_classes_FangHui.data"))
    CFL_report =  joblib.load(os.path.join(root_dir,dataset_name,"CFL","eval_classes_FangHui.data"))
    Dummy_report =  joblib.load(os.path.join(root_dir,dataset_name,"Dummy","eval_classes_FangHui.data"))
    MCCP_list = get_class_avg_precision(MCCP_report)
    HMR_list = get_class_avg_precision(HMR_report)
    CFL_list = get_class_avg_precision(CFL_report)
    Dummy_list = []
    for class_i in range(global_class_num):
        Dummy_list.append(Dummy_report[str(class_i)]["recall"])
    fig = plt.figure(figsize=(7,4))  
    x_1 = list(range(len(class_list)))
    bar_width = 0.15 # 柱子宽度
    plt.bar(x_1, MCCP_list, width=bar_width, label="MCCP", fc="red")
    # 第二个柱子的位置
    x_2 = list(range(len(class_list)))
    for i in range(len(x_2)):
        x_2[i] = x_2[i]+bar_width
    plt.bar(x_2, HMR_list, width=bar_width, label="HMR", fc="green") # tick_label = class_list
    # 第三个柱子的位置
    x_3 = list(range(len(class_list)))
    for i in range(len(x_3)):
        x_3[i] = x_3[i]+2*bar_width
    plt.bar(x_3, CFL_list, width=bar_width, label="CFL", fc="blue")
    # 第四个柱子的位置
    x_4 = list(range(len(class_list)))
    for i in range(len(x_4)):
        x_4[i] = x_4[i]+3*bar_width
    plt.bar(x_4, Dummy_list, width=bar_width, label="Dummy", fc="orange")
    plt.xticks(rotation=-15)  
    ticks_positions = [x+0.5*bar_width for x in x_2]
    plt.xticks(ticks_positions, class_list)
    # plt.ylim(0.4, 1.0)  
    plt.legend()
    plt.tight_layout()
    plt.show()
    save_dir = f"exp_image/{dataset_name}"
    file_name = "case_study_classes.pdf"
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)
    print("draw_case_study_bar() successfully!")

def draw_overlap_unqiue_avg_improve_line(config):

    dataset_name = config["dataset_name"]
    unique_trained_data = joblib.load(f"exp_data/{dataset_name}/retrainResult/percent/OurCombin/train_unique_v3.data")
    overlap_trained_data = joblib.load(f"exp_data/{dataset_name}/retrainResult/percent/OurCombin/train_overlap_v3.data")
    initAcc = joblib.load(f"exp_data/{dataset_name}/initAcc.data") 
    overlap_initAcc = initAcc["overlap_initAcc"]["accuracy"]
    unique_initAcc = initAcc["unique_initAcc"]["accuracy"]
    
    unique_y = []
    overlap_y = []
    percent_list = [1,3,5,10,15,20]
    for percent in percent_list:
        avg_u = sum(unique_trained_data["train"][percent])/len(unique_trained_data["train"][percent])
        u_improve = avg_u - unique_initAcc
        u_improve = round(u_improve,4)
        unique_y.append(u_improve)

        avg_o = sum(overlap_trained_data["train"][percent])/len(overlap_trained_data["train"][percent])
        o_improve = avg_o - overlap_initAcc
        o_improve = round(o_improve,4)
        overlap_y.append(o_improve)
    plt.figure(figsize=(3,3))

    x_list = ["1%","3%","5%", "10%", "15%", "20%"]
    # 画线
    line_1 = plt.plot(x_list, unique_y, label = "unique", color = "red", marker = "x")
    line_2 = plt.plot(x_list, overlap_y, label = "overlapping", color = "green", marker = "o")
    # 画网格
    plt.grid()
    # 图例
    plt.legend()
    # 坐标轴说明
    plt.xlabel("Sampling ratio", fontproperties = "Times New Roman", fontsize = 12)
    plt.ylabel("Improved accuracy", fontproperties = "Times New Roman", fontsize = 12)
    plt.xticks(fontproperties = "Times New Roman", fontsize = 11)
    plt.xticks(fontproperties = "Times New Roman", fontsize = 11)
    save_dir = f"exp_image/{dataset_name}"
    file_name = "discussion_improve.pdf"
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path, bbox_inches = 'tight', pad_inches = 0.1, dpi=600)
    print("draw_overlap_unqiue_avg_improve_line() successfully!")

def draw_unique_overlap_avg_line():
    dataset_name = config["dataset_name"]
    unique_data = joblib.load(f"exp_data/{dataset_name}/retrainResult/percent/OurCombin/train_unique_v3.data")
    overlap_data = joblib.load(f"exp_data/{dataset_name}/retrainResult/percent/OurCombin/train_overlap_v3.data")
    percent_list = [1,3,5,10,15,20]

    unique_y = []
    overlap_y = []
    for percent in percent_list:
        avg_u = sum(unique_data["train"][percent])/len(unique_data["train"][percent])
        avg_u = round(avg_u,4)
        unique_y.append(avg_u)

        avg_o = sum(overlap_data["train"][percent])/len(overlap_data["train"][percent])
        avg_o = round(avg_o,4)
        overlap_y.append(avg_o)
    x_list = ["1%","3%","5%", "10%", "15%", "20%"]
    # 画线
    line_1 = plt.plot(x_list, unique_y, label = "unique", color = "red", marker = "x")
    line_2 = plt.plot(x_list, overlap_y, label = "overlap", color = "green", marker = "o")
    # 画网格
    plt.grid()
    # 图例
    plt.legend()
    # 坐标轴说明
    plt.xlabel("Sampling ratio")
    plt.ylabel("Accuracy")

    save_dir = f"exp_image/{dataset_name}"
    file_name = "unique_overlap_avg_line.pdf"
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)
    print("draw_unique_overlap_avg_line successfully!")

def draw_unique_overlap_var_line(config):
    dataset_name = config["dataset_name"]
    unique_data = joblib.load(f"exp_data/{dataset_name}/retrainResult/percent/OurCombin/train_unique_v3.data")
    overlap_data = joblib.load(f"exp_data/{dataset_name}/retrainResult/percent/OurCombin/train_overlap_v3.data")
    percent_list = [1,3,5,10,15,20]

    unique_y = []
    overlap_y = []
    for percent in percent_list:
        var_u = np.var(unique_data["train"][percent])
        var_u = round(var_u,4)
        unique_y.append(var_u)

        var_o = np.var(overlap_data["train"][percent])
        var_o = round(var_o,4)
        overlap_y.append(var_o)

    x_list = ["1%","3%","5%", "10%", "15%", "20%"]
    # 画线
    line_1 = plt.plot(x_list, unique_y, label = "unique", color = "red", marker = "x")
    line_2 = plt.plot(x_list, overlap_y, label = "overlap", color = "green", marker = "o")
    # 画网格
    plt.grid()
    # 图例
    plt.legend()
    # 坐标轴说明
    plt.xlabel("Sampling ratio")
    plt.ylabel("Variance of accuracy")

    save_dir = f"exp_image/{dataset_name}"
    file_name = "unique_overlap_var_line.pdf"
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)
    print("draw_unique_overlap_var_line successfully!")

def draw_slope(config):
    dataset_name = config["dataset_name"]
    our_data = joblib.load(f"exp_data/{dataset_name}/retrainResult/percent/OurCombin/train_acc_v3.data")
    init_acc = our_data["init_acc"]
    repeat_list = [1,3,5,10,15,20]
    mean_list = []
    mean_list.append(init_acc)
    for repeat in repeat_list:
        mean_list.append(round(np.mean(our_data["train"][repeat]),4))
    x_list = ["1%","3%","5%", "10%", "15%", "20%"]
    # 计算mean_list增长率
    mean_list = np.array(mean_list)
    slope = np.diff(mean_list) / mean_list[:-1]
    print(slope)

def draw_heatmap():
    spearman_corr = pd.read_csv("exp_data/all/spearman_corr.csv", index_col=0)
    # mask = np.zeros_like(spearman_corr, dtype=np.bool)  # 定义一个大小一致全为零的矩阵  用布尔类型覆盖原来的类型
    # mask[np.triu_indices_from(mask)]= True  #返回矩阵的上三角，并将其设置为true
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # sns.set(font_scale=1.2)
    # plt.figure(figsize=(9,9))
    f, ax = plt.subplots(figsize = (9, 7))
    sns.heatmap(spearman_corr,
                cmap="RdBu_r",
                square = True,
                # mask=mask, #只显示为true的值
                center=0,
                linewidths=.5, # 控制小方格间距
                cbar_kws={"shrink": .5},
                annot=True     #底图带数字 True为显示数字
                )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=360)
    save_dir = "exp_image/all"
    file_name = "corr.pdf"
    file_path = os.path.join(save_dir, file_name)
    # plt.show()
    f.savefig(file_path, dpi=800, bbox_inches="tight")

def draw_stable_bar(config):
    root_dir = "/data2/mml/overlap_v2_datasets/"
    dataset_name = config["dataset_name"]
    MCCP_trueOrFalse_ans = joblib.load(os.path.join(root_dir, dataset_name, "MCCP", "eval_TrueOrFalse_list_FangHui.data"))
    HMR_trueOrFalse_ans = joblib.load(os.path.join(root_dir, dataset_name, "HMR", "eval_TrueOrFalse_list_FangHui.data"))
    data = defaultdict(list)
    sample_rate_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    base_rate = sample_rate_list[0]
    for repeat_i in range(10):
        MCCP_trueOrFalse_list = MCCP_trueOrFalse_ans[base_rate][repeat_i]
        HMR_trueOrFalse_list = HMR_trueOrFalse_ans[base_rate][repeat_i]
        MCCP_acc = sum(MCCP_trueOrFalse_list)/len(MCCP_trueOrFalse_list)
        HMR_acc = sum(HMR_trueOrFalse_list)/len(MCCP_trueOrFalse_list)
        base_dif_i_list = []
        for i in range(len(MCCP_trueOrFalse_list)):
            MCCP_trueOrFalse = MCCP_trueOrFalse_list[i]
            HMR_trueOrFalse = HMR_trueOrFalse_list[i]
            if MCCP_trueOrFalse == True and HMR_trueOrFalse == False:
                base_dif_i_list.append(i)
        intersection = set(base_dif_i_list)
        for sample_rate in sample_rate_list:
            MCCP_trueOrFalse_list = MCCP_trueOrFalse_ans[sample_rate][repeat_i]
            HMR_trueOrFalse_list = HMR_trueOrFalse_ans[sample_rate][repeat_i]
            dif_i_list = []
            for i in range(len(MCCP_trueOrFalse_list)):
                MCCP_trueOrFalse = MCCP_trueOrFalse_list[i]
                HMR_trueOrFalse = HMR_trueOrFalse_list[i]
                if MCCP_trueOrFalse == True and HMR_trueOrFalse == False:
                    dif_i_list.append(i)
            intersection = intersection.intersection(set(dif_i_list))
        win_rate = len(intersection)/len(MCCP_trueOrFalse_list)
        win_acc = MCCP_acc - HMR_acc
        print("jflaj")
            # percent = len(intersection)/len(base_dif_i_list)
            # data[repeat_i].append(percent)
        
    matrix = np.array([])
    for repeat_i in range(10):
        matrix = np.append(matrix,data[repeat_i])
    print(matrix)

def draw_classes_improve(config):
    root_dir = "/data2/mml/overlap_v2_datasets"
    dataset_name = config["dataset_name"]
    classes_acc_record = joblib.load(os.path.join(root_dir, dataset_name, "MCCP", "eval_classes_FangHui.data"))
    base_classes_acc = joblib.load(os.path.join(root_dir, dataset_name, "MCCP", "init_merged_test_classes_acc.data"))
    local_to_global_A = joblib.load(config["local_to_global_party_A_path"])
    local_to_global_B = joblib.load(config["local_to_global_party_B_path"])
    global_A = list(local_to_global_A.values())
    global_B = list(local_to_global_B.values())
    global_index_list = sorted(list(set(list(set(global_A+global_B)))))
    global_overlap = sorted(list(set(global_A).intersection(set(global_B))))

    # sample_rate_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    sample_rate = 0.2
    improve_list = []
    for global_idx in global_index_list:
        MCCP_base_precision = base_classes_acc[str(global_idx)]["recall"]
        avg_improve = 0
        for repeat_i in range(10):
            precision = classes_acc_record[sample_rate][repeat_i][str(global_idx)]["recall"]
            improve = precision - MCCP_base_precision
            avg_improve += improve
        avg_improve = avg_improve/10
        improve_list.append(avg_improve)
    fig = plt.figure(figsize=(4,3))  
    x = global_index_list
    colors = []
    for global_idx in global_index_list:
        if global_idx in global_overlap:
            colors.append("green")
        else:
            colors.append("red")
    bar_width = 0.5 # 柱子宽度
    tick_label = [f"Class_{str(class_idx)}" for class_idx in global_index_list]
    plt.bar(x, improve_list, width=bar_width, label="MCCP", tick_label = tick_label, color=colors)
    plt.grid()
    plt.xticks(rotation=-15)
    save_dir = f"exp_image/{dataset_name}"
    save_file_name = f"classes_improve_{str(sample_rate)}.pdf"
    save_file_path = os.path.join(save_dir,save_file_name)
    plt.savefig(save_file_path,bbox_inches='tight')

def draw_Overlap_unique_improve(config):
    root_dir = "/data2/mml/overlap_v2_datasets"
    dataset_name = config["dataset_name"]
    overlap_ans = joblib.load(os.path.join(root_dir, dataset_name, "MCCP", "eval_ans_Overlap_FangHui.data"))
    unique_ans = joblib.load(os.path.join(root_dir, dataset_name, "MCCP", "eval_ans_Unique_FangHui.data"))
    overlap_base = joblib.load(os.path.join(root_dir, dataset_name, "MCCP", "init_overlap_test_acc.data"))
    unique_base = joblib.load(os.path.join(root_dir, dataset_name, "MCCP", "init_unique_test_acc.data"))
    sample_rate_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    overlap_improve_list = []
    unique_improve_list = []
    for sample_rate in sample_rate_list:
        avg_overlap_improve = 0
        avg_unique_improve = 0
        for repeat_i in range(10):
            overlap_acc =  overlap_ans[sample_rate][repeat_i]
            overlap_improve = overlap_acc - overlap_base
            avg_overlap_improve += overlap_improve
            unique_acc =  unique_ans[sample_rate][repeat_i]
            unique_improve = unique_acc - unique_base
            avg_unique_improve += unique_improve
        avg_unique_improve /= 10
        avg_overlap_improve /= 10
        overlap_improve_list.append(avg_overlap_improve)
        unique_improve_list.append(avg_unique_improve)
    plt.figure(figsize=(3,3))

    x_list = ["1%","3%","5%", "10%", "15%", "20%"]
    # 画线
    line_1 = plt.plot(x_list, unique_improve_list, label = "unique", color = "red", marker = "x")
    line_2 = plt.plot(x_list, overlap_improve_list, label = "overlapping", color = "green", marker = "o")
    # 画网格
    plt.grid()
    # 图例
    plt.legend()
    # 坐标轴说明
    plt.xlabel("Sampling ratio", fontproperties = "Times New Roman", fontsize = 12)
    plt.ylabel("Improved accuracy", fontproperties = "Times New Roman", fontsize = 12)
    plt.xticks(fontproperties = "Times New Roman", fontsize = 11)
    plt.xticks(fontproperties = "Times New Roman", fontsize = 11)
    save_dir = f"exp_image/{dataset_name}"
    file_name = "discussion_improve_new.pdf"
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path, bbox_inches = 'tight', pad_inches = 0.1, dpi=600)
    print("draw_Overlap_unique_improve() successfully!")


    pass
if __name__ == "__main__":
    # 全局变量区
    config = animal_2_config
    # local_to_global_A = joblib.load(config["local_to_global_party_A_path"])
    # local_to_global_B = joblib.load(config["local_to_global_party_B_path"])
    # print(local_to_global_A)
    # print(local_to_global_B)
    # draw_overlap_unqiue_avg_improve_line(config)
    # draw_slope(config)
    # draw_overlap_unqiue_initAcc_bar()
    # draw_overlap_initAcc_bar()
    # draw_unique_initAcc_bar()
    # draw_merged_initAcc_bar()
    # draw_case_study_bar()
    # draw_unique_overlap_avg_line()
    # draw_unique_overlap_var_line()
    # draw_box()
    # draw_line_main(config)
    # draw_line_main_2(config)
    # draw_line_main_ML(config)
    # draw_heatmap()
    # draw_case_study(config)
    # draw_stable_bar(config)
    draw_classes_improve(config)
    # draw_Overlap_unique_improve(config)
    pass