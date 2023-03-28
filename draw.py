'''
画图文件
'''
import joblib
from matplotlib import pyplot as plt
import os

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
    fig = plt.figure(figsize=(10, 4), dpi=800)
    # 激活区域
    plt.subplot(1,2,1)
    plt.xticks(x_list)
    # 画线
    line_1 = plt.plot(x_list, ourCombin_list, label = "ourCombin", color = "red", marker = "x")
    line_2 = plt.plot(x_list, hmr_list, label = "HMR", color = "green", marker = "o")
    # 画网格
    plt.grid()
    # 画水平线
    plt.axhline(y_data["OurCombin_base_acc"],color='lightcoral',ls='--')
    plt.axhline(y_data["HMR_base_acc"],color='springgreen',ls='--')
    # 图例
    plt.legend()
    # 坐标轴说明
    plt.xlabel("num of sampling")
    plt.ylabel("Accuracy")
    # 图题目
    plt.title("Accuracy of Knowledge Amalgamation")
    
    plt.subplot(1,2,2)
    plt.xticks(x_list)
    line_3 = plt.plot(x_list, cfl_list, label = "CFL", color = "blue", marker = 's')
    plt.grid()
    plt.axhline(y_data["CFL_base_acc"],color='cornflowerblue',ls='--')
    plt.legend()
    plt.xlabel("num of sampling")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of Knowledge Amalgamation")
    return fig

def draw_line_main():
    dataset_name = "car_body_style"
    sample_method = "num"
    resulst_file_name = "reTrain_acc.data"
    # x轴(x_list)
    x_list = ["1%","3%","5%", "10%", "15%", "20%", "50%", "80", "100%"]
    x_list = [30,60,90,120,150,180]
    # 加载数据
    OurCombin = joblib.load(f"exp_data/{dataset_name}/retrainResult/{sample_method}/OurCombin/{resulst_file_name}")
    HMR = joblib.load(f"exp_data/{dataset_name}/retrainResult/{sample_method}/HMR/{resulst_file_name}")
    CFL = joblib.load(f"exp_data/{dataset_name}/retrainResult/{sample_method}/CFL/{resulst_file_name}")
    # 获得y_list
    ourCombin_list  = get_y_list(OurCombin)
    hmr_list = get_y_list(HMR)
    cfl_list = get_y_list(CFL)
    # 封装字典
    y_data = {}
    y_data["OurCombin"] = ourCombin_list
    y_data["HMR"] = hmr_list
    y_data["CFL"] = cfl_list
    y_data["OurCombin_base_acc"] = OurCombin["base_acc"]
    y_data["HMR_base_acc"] = HMR["base_acc"]
    y_data["CFL_base_acc"] = CFL["base_acc"]
    # 画图
    fig = draw_lines(x_list, y_data)
    # 保存图片
    save_dir = f"exp_image/{dataset_name}"
    file_name = f"our_hmr_cfl_{sample_method}_v3.png"
    file_path = os.path.join(save_dir, file_name)
    fig.savefig(file_path)


if __name__ == "__main__":
    draw_line_main()