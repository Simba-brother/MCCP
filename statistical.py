import joblib
import os
from DatasetConfig_2 import config as config_2
from scipy import stats
from cliffs_delta import cliffs_delta
from utils import makedir_help

    
def stat_WTL(root_dir, config, save_path):
    root_dir = "/data2/mml/overlap_v2_datasets"
    dataset_name_list = config["dataset_name_list"]
    method_list = ["HMR", "CFL", "Dummy"]
    sample_rate_list = config["sample_rate_list"]
    '''
    ans = {
        "CFL":{
            "0.01":{
                "car_body_style":W,
                "flower_2":T
            }
        }
    }
    '''
    ans = {}
    for method in method_list:
        ans[method] = {}
        for sample_rate in sample_rate_list:
            ans[method][sample_rate] = {}
            for dataset_name in dataset_name_list:
                ans[method][sample_rate][dataset_name] = None

    for dataset_name in dataset_name_list:
        MCCP_eval_data = joblib.load(os.path.join(root_dir, dataset_name, "MCCP", "eval_ans_FangHui.data"))
        HMR_eval_data = joblib.load(os.path.join(root_dir, dataset_name, "HMR", "eval_ans_FangHui.data"))
        CFL_eval_data = joblib.load(os.path.join(root_dir, dataset_name, "CFL", "eval_ans_FangHui.data"))
        Dummy_eval_data = joblib.load(os.path.join(root_dir, dataset_name, "Dummy", "eval_ans_FangHui.data"))
        for sample_rate in sample_rate_list:
            MCCP_eval_list = MCCP_eval_data[sample_rate]
            HMR_eval_list = HMR_eval_data[sample_rate]
            CFL_eval_list = CFL_eval_data[sample_rate]
            Dummy_eval_list = [Dummy_eval_data]*len(MCCP_eval_list)
            MCCP_eval_list_sorted = sorted(MCCP_eval_list)
            HMR_eval_lists_sorted = sorted(HMR_eval_list)
            CFL_eval_list_sorted = sorted(CFL_eval_list)
            Dummy_eval_list_sorted = sorted(Dummy_eval_list)
            # 计算p值和Cliff’s delta
            # MCCP-HMR
            p_value = stats.wilcoxon(MCCP_eval_list, HMR_eval_list).pvalue
            dela,info = cliffs_delta(MCCP_eval_list_sorted, HMR_eval_lists_sorted)
            if p_value < 0.05 and dela > 0.147:
                ans["HMR"][sample_rate][dataset_name] = "W" # win
            elif p_value < 0.05 and dela < 0.147:
                ans["HMR"][sample_rate][dataset_name] = "L" # loss
            else:
                ans["HMR"][sample_rate][dataset_name] = "T" # Tie
            # MCCP-CFL
            p_value = stats.wilcoxon(MCCP_eval_list, CFL_eval_list).pvalue
            dela,info = cliffs_delta(MCCP_eval_list_sorted, CFL_eval_list_sorted)
            if p_value < 0.05 and dela > 0.147:
                ans["CFL"][sample_rate][dataset_name] = "W" # win
            elif p_value < 0.05 and dela < 0.147:
                ans["CFL"][sample_rate][dataset_name] = "L" # loss
            else:
                ans["CFL"][sample_rate][dataset_name] = "T" # Tie
            # MCCP-Dummy
            p_value = stats.wilcoxon(MCCP_eval_list, Dummy_eval_list).pvalue
            dela,info = cliffs_delta(MCCP_eval_list_sorted, Dummy_eval_list_sorted)
            if p_value < 0.05 and dela > 0.147:
                ans["Dummy"][sample_rate][dataset_name] = "W" # win
            elif p_value < 0.05 and dela < 0.147:
                ans["Dummy"][sample_rate][dataset_name] = "L" # loss
            else:
                ans["Dummy"][sample_rate][dataset_name] = "T" # Tie
    joblib.dump(ans,save_path)
    print(f"save_path:{save_path}")
    print("stat_WTL end")
    return ans

if __name__ == "__main__":

    root_dir = "/data2/mml/overlap_v2_datasets"
    save_dir = os.path.join("exp_data", "all")
    makedir_help(save_dir)
    save_file_name = "WTL.data"
    save_path = os.path.join(save_dir, save_file_name)
    ans = stat_WTL(root_dir, config_2, save_path)


    