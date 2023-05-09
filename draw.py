'''
画图文件
'''
import joblib
from matplotlib import pyplot as plt
import os
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
    hmr_list = y_data["HMR"]
    cfl_list = y_data["CFL"]
    # 画布
    fig = plt.figure(figsize=(15, 5.5), dpi=600)
    # 激活区域
    plt.subplot(1,2,1)
    # plt.xticks(x_list)
    # 画线
    line_1 = plt.plot(x_list, ourCombin_list, label = "ourCombin", color = "red", marker = "x")
    line_2 = plt.plot(x_list, hmr_list, label = "HMR", color = "green", marker = "o")
    # 画网格
    plt.grid()
    # 画水平线
    plt.axhline(y_data["OurCombin_base_acc"],color='lightcoral',ls='--')
    plt.axhline(y_data["HMR_base_acc"],color='springgreen',ls='--')
    plt.axhline(y_data["Dummy_base_acc"], color = "orange", ls="solid", marker="^", label="Dummy")
    # 图例
    plt.legend()
    # 坐标轴说明
    plt.xlabel("percent of sampling", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.tick_params(labelsize=15)  #刻度字体大小13
    # 图题目
    # plt.title("Accuracy of Knowledge Amalgamation")
    
    
    plt.subplot(1,2,2)
    # plt.xticks(x_list)
    line_3 = plt.plot(x_list, cfl_list, label = "CFL", color = "blue", marker = 's')
    plt.grid()
    plt.axhline(y_data["CFL_base_acc"],color='cornflowerblue',ls='--')
    plt.axhline(y_data["Dummy_base_acc"], color = "orange", ls="solid", marker="^", label="Dummy")
    plt.legend()
    plt.xlabel("percent of sampling", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    # plt.title("Accuracy of Knowledge Amalgamation")
    plt.tick_params(labelsize=15)  #刻度字体大小13
    return fig

def draw_truncation_line(x_list, y_data):
    ourCombin_list = y_data["OurCombin"]
    hmr_list = y_data["HMR"]
    cfl_list = y_data["CFL"]
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(4, 4), dpi=600)  # 绘制两个子图
    plt.xlabel("percent of sampling")
    plt.ylabel("Accuracy")

    ax1.xaxis.tick_top()
    ax1.xaxis.set_visible(True)
    ax2.xaxis.set_visible(True)
    plt.subplots_adjust(wspace=0,hspace=0.08) # 设置子图间距


    ax2.plot(x_list, cfl_list, label = "CFL", color = "blue", marker = 's')


    ax1.plot(x_list, ourCombin_list, label = "ourCombin", color = "red", marker = "x")   # 绘制折线
    ax1.plot(x_list, hmr_list, label = "HMR", color = "green", marker = "o")
    ax1.plot(x_list, cfl_list, label = "CFL", color = "blue", marker = 's')
    ax1.axhline(y_data["Dummy_base_acc"], color = "orange", ls="solid", marker="^", label="Dummy")
    ax1.set_ylim(0.73, 0.85) # 设置纵坐标范围

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

def draw_line_main():
    dataset_name = "car_body_style"
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
    file_name = f"our_hmr_cfl_dummy_{sample_method}_v4.pdf"
    file_path = os.path.join(save_dir, file_name)
    fig.savefig(file_path)

def draw_box():
    dataset_name = config["dataset_name"]
    labels = '1%', '3%', '5%', '10%', '15%', '20%'
    train_acc = joblib.load(f"exp_data/{dataset_name}/retrainResult/percent/OurCombin/train_acc_v3.data")
    A = train_acc["train"][1]
    B = train_acc["train"][3]
    C = train_acc["train"][5]
    D = train_acc["train"][10]
    E = train_acc["train"][15]
    F = train_acc["train"][20]
    data = [A, B, C, D, E, F]
    plt.grid(True)  # 显示网格
    plt.boxplot(data,
                medianprops={'color': 'red', 'linewidth': '1.5'}, # 设置中位数的属性，如线的类型、粗细等；
                showmeans=True,
                meanline=False,
                # meanline=True,
                # meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10}, # 设置异常值的属性，如异常点的形状、大小、填充色等
                labels=labels)
    # plt.yticks(np.arange(0.4, 0.81, 0.1))
    plt.xlabel("Sampling ratio")
    plt.ylabel("Accuracy")
    plt.show()
    save_dir = f"exp_image/{dataset_name}"
    file_name = "box_full_classification_stability.pdf"
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)
    print("draw_box successfully!")

# 全局变量区
config = weather_config

if __name__ == "__main__":
    # draw_box()
    # draw_line_main()
    pass