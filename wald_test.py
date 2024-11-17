'''
参考文献 Training Data Debugging for the Fairness of Machine Learning Software,(ICSE'22),Linghan Meng,Huiyan Li)
技术参考：
(1):https://zhuanlan.zhihu.com/p/22692029
(2):https://www.datascienceconcepts.com/tutorials/python-programming-language/omitted-variable-bias-wald-test-in-python/
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def get_rank(data_list,reverse_flag=False):
    rank = []
    data_list_sorted = sorted(data_list,reverse=reverse_flag)
    for data in data_list:
        rank.append(data_list_sorted.index(data))
    return rank


def calcu(x_list,y_list):
    '''
    x_list:是自变量
    y_list:是因变量
    '''
    x_list =  sm.add_constant(x_list)
    model = sm.OLS(y_list, x_list) # 此处注意因变量在前。
    results = model.fit()
    wald_test = results.wald_test('x1 = 0')  # 检验X的系数是否为0
    return results, wald_test

def construct_dataFrame():
    Acc_diff_list = [0.0153,0.1326,0.2828,0.0386,0.17,0.0513,0.0624,-0.0261,0.384,0.2385,0.0272,0.2793,0.0381,0.1434,-0.079,0.7774,-0.0851,0.0975]
    Cosine_similarity_list_2 = [0.6465,0.6465,0.7452,0.7452,0.6724,0.6724,0.7976,0.7976,0.5915,0.5915,0.5517,0.5517,0.7813,0.7813,0.7524,0.7524,0.7759,0.7759]
    cosine_similarity_list_3 = [0.8061,0.7667,0.8708,0.8663,0.7862,0.8322,0.8721,0.8999,0.7183,0.7757,0.7519,0.6951,0.8692,0.8896,0.8704,0.862,0.9044,0.8565]
    


    Jaccard_list_2 = [0.51,0.51,0.78,0.78,0.69,0.69,0.67,0.67,0.58,0.58,0.53,0.53,0.59,0.59,0.74,0.74,0.92,0.92]
    ACC_A = [0.7873,0.6194,0.9349,0.8979,0.945,0.84,0.7917,0.9583,0.9512,0.7233,0.8008,0.6262,0.8092,0.7763,0.89,0.21,0.9068,0.9025]
    ACC_B = [0.772,0.752,0.6521,0.9365,0.775,0.8903,0.7293,0.9322,0.5672,0.9618,0.7736,0.9055,0.7711,0.9197,0.969,0.9874,0.9919,1]
    data = {
        "Acc_diff":Acc_diff_list,
        "Cosine_similarity_2":Cosine_similarity_list_2,
        "Cosine_similarity_3":cosine_similarity_list_3,
        "Jaccard_2":Jaccard_list_2,
        "ACC_A":ACC_A,
        "ACC_B":ACC_B
    }
    df = pd.DataFrame(data)
    return df

def main_2():
    '''
    另外一种api计算方式
    '''
    df = construct_dataFrame()
    #fit multiple linear regression model
    results = smf.ols('Acc_diff ~ Cosine_similarity_2', df).fit()
    #view regression model summary
    print(results.summary())
    print(results.wald_test('Cosine_similarity_2 = 0'))
    print("=*="*50)
    results_2 = smf.ols('Acc_diff ~ Cosine_similarity_3', df).fit()
    print(results_2.summary())
    #perform Wald Test to determine if 'hp' and 'cyl' coefficients are both zero
    print(results_2.wald_test('Cosine_similarity_3 = 0'))

def main():
    A_diff_list = [0.0153,0.2828,0.17,0.0624,0.384,0.0272,0.0381,0.079,0.0851]
    B_diff_list = [0.1326,0.0386,0.0513,0.0261,0.2385,0.2793,0.1434,0.7774,0.0975]
    ACC_diff_list = [0.0153,0.1326,0.2828,0.0386,0.17,0.0513,0.0624,-0.0261,0.384,0.2385,0.0272,0.2793,0.0381,0.1434,-0.079,0.7774,-0.0851,0.0975]
    AB_dif_abs = []
    cosine_similarity_list = [0.6465,0.7452,0.6724,0.7976,0.5915,0.5517,0.7813,0.7524,0.7759]
    cosine_similarity_list_2 = [0.6465,0.6465,0.7452,0.7452,0.6724,0.6724,0.7976,0.7976,0.5915,0.5915,0.5517,0.5517,0.7813,0.7813,0.7524,0.7524,0.7759,0.7759]
    cosine_similarity_list_3 = [0.8061,0.7667,0.8708,0.8663,0.7862,0.8322,0.8721,0.8999,0.7183,0.7757,0.7519,0.6951,0.8692,0.8896,0.8704,0.862,0.9044,0.8565]
    jaccard_list = [0.51,0.78,0.69,0.67,0.58,0.53,0.59,0.74,0.92]
    jaccard_list_2 = [0.51,0.51,0.78,0.78,0.69,0.69,0.67,0.67,0.58,0.58,0.53,0.53,0.59,0.59,0.74,0.74,0.92,0.92]

    results_AandCos,wald_test_AandCos = calcu(x_list=cosine_similarity_list,y_list=A_diff_list)
    results_BandCos,wald_test_BandCos = calcu(x_list=cosine_similarity_list,y_list=B_diff_list)
    results_AandB,wald_test_AandB = calcu(x_list=B_diff_list,y_list=A_diff_list)
    results_ACC_diffandCos2,wald_test_ACC_diffandCos2 = calcu(x_list=cosine_similarity_list_2,y_list=ACC_diff_list)
    results_ACC_diffandJaccard2,wald_test_ACC_diffandJaccrad2 = calcu(x_list=jaccard_list_2,y_list=ACC_diff_list)
    results_JaccradandcCos,wald_test_JaccradandcCos = calcu(x_list=cosine_similarity_list,y_list=jaccard_list)


    print("AandCos:")
    print(results_AandCos.summary())
    print("AandCos回归系数:")
    print(results_AandCos.params)
    print("AandCos回归系数P值:")
    print(results_AandCos.pvalues)
    print("wald_test_AandCos:")
    print(wald_test_AandCos)

    print("*"*100)

    print("BandCos:")
    print(results_BandCos.summary())
    print("BandCos回归系数:")
    print(results_AandCos.params)
    print("BandCos回归系数P值:")
    print(results_BandCos.pvalues)
    print("wald_test_BandCos:")
    print(wald_test_BandCos)

    print("*"*100)

    print("AandB:")
    print(results_AandB.summary())
    print("AandB回归系数:")
    print(results_AandB.params)
    print("AandB回归系数P值:")
    print(results_AandB.pvalues)
    print("wald_test_AandB:")
    print(wald_test_AandB)

    print("*"*100)

    print("ACC_diffandJaccard2:")
    print(results_ACC_diffandJaccard2.summary())
    print("ACC_diffandJaccard2回归系数:")
    print(results_ACC_diffandJaccard2.params)
    print("ACC_diffandJaccard2回归系数P值:")
    print(results_ACC_diffandJaccard2.pvalues)
    print("wald_test_ACC_diffandJaccrad2:")
    print(wald_test_ACC_diffandJaccrad2)
    

    print("*"*100)

    print("JaccradandcCos:")
    print(results_JaccradandcCos.summary())
    print("JaccradandcCos回归系数:")
    print(results_JaccradandcCos.params)
    print("JaccradandcCos回归系数P值:")
    print(results_JaccradandcCos.pvalues)
    print("wald_test_JaccradandcCos:")
    print(wald_test_JaccradandcCos)

    print("*"*100)

    print("ACC_difandCos2:")
    print(results_ACC_diffandCos2.summary())
    print("ACC_difandCos2回归系数:")
    print(results_ACC_diffandCos2.params)
    print("ACC_difandCos2回归系数P值:")
    print(results_ACC_diffandCos2.pvalues)
    print("wald_test_ACC_diffandCos2:")
    print(wald_test_ACC_diffandCos2)

    '''
    可视化
    y_fitted = results_AandCos.fittedvalues
    fig, ax = plt.subplots(figsize=(8,6))
    # 画出原始数据
    ax.plot(cosine_similarity_list, A_diff_list, 'o', label='data')
    # 画出拟合数据，图像为红色带点间断线
    ax.plot(cosine_similarity_list, y_fitted, 'r--.',label='OLS')
    # 放置注解。
    ax.legend(loc='best')
    plt.savefig("exp_image/wald/AandCos.png",dpi=800)
    '''

'''
实例代码
def temp():
    # 生成模拟数据
    np.random.seed(0)
    X = np.random.rand(100, 1) * 10   # 100个随机点
    y = 2 * X.squeeze() + np.random.randn(100) * 2  # 目标值

    # 将X添加常数项,importent
    X = sm.add_constant(X)

    # 构建线性回归模型
    model = sm.OLS(y, X)
    results = model.fit()

    # 打印模型摘要
    print(results.summary())

    # 进行Wald检验
    wald_test = results.wald_test('x1 = 0')  # 检验X的系数是否为0
    print(wald_test)
'''
if __name__ == "__main__":
   # main()
   main_2()

