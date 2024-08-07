import numpy as np
import mpyc
import gmpy2
import sys
import time
import matplotlib.pyplot as plt
import random
from data_process import data_process



# 简单分布式梯度下降算法
# 以简单的经济调度为例ED

def ED_DGD_main():

    area_num = 3
    D1, P_max_1, P_min_1, coe_c_1, epison_1, gen_num1 = data_process(1)
    D2, P_max_2, P_min_2, coe_c_2, epison_2, gen_num2 = data_process(2)
    D3, P_max_3, P_min_3, coe_c_3, epison_3, gen_num3 = data_process(3)

    #分布式梯度更新
    """
    这里为区域1来定义其相关变量
    """
    # 定义对偶变量
    f1 = np.zeros((1, 24))  # 需要共享的数据以及需要隐私保护
    f1_share = np.zeros((area_num, 24))  # 为方便后续共同计算
    # f1_share[0, :] = f1
    # 定义对偶变量的梯度
    d_f1 = np.zeros((1, 24))
    d_f1 = d_f1 + D1
    df1_share = np.zeros((area_num, 24))
    df1_share[0, :] = d_f1
    # 定义原始变量即功率
    p11 = np.zeros((1, 24))
    p12 = np.zeros((1, 24))
    p13 = np.zeros((1, 24))
    p1 = [p11, p12, p13]
    # 信息图矩阵
    Bij_1 = np.array([[0.5, 0.3, 0.2]])  # 需要共享的数据以及需要隐私保护
    Bij_share_1 = np.zeros((area_num, area_num))
    Bij_share_1[0, 1:3] = Bij_1[0, 1:3]


    """
    这里为区域2来定义其相关变量
    """
    f2 = np.zeros((1, 24))  # 需要共享的数据以及需要隐私保护
    f2_share = np.zeros((area_num, 24))  # 为方便后续共同计算
    # f2_share[1, :] = f2
    # 定义对偶变量的梯度
    d_f2 = np.zeros((1, 24))
    d_f2 = d_f2 + D2
    df2_share = np.zeros((area_num, 24))
    df2_share[1, :] = d_f2
    # 定义原始变量即功率
    p21 = np.zeros((1, 24))
    p22 = np.zeros((1, 24))
    p2 = [p21, p22]
    # 信息图矩阵
    Bij_2 = np.array([[0.3, 0.5, 0.2]])  # 需要共享的数据以及需要隐私保护
    Bij_share_2 = np.zeros((area_num, area_num))
    Bij_share_2[1, 0] = Bij_2[0, 0]
    Bij_share_2[1, 2] = Bij_2[0, 2]

    """
    这里为区域3来定义其相关变量
    """
    f3 = np.zeros((1, 24))  # 需要共享的数据以及需要隐私保护
    f3_share = np.zeros((area_num, 24))  # 为方便后续共同计算
    # f2_share[1, :] = f2
    # 定义对偶变量的梯度
    d_f3 = np.zeros((1, 24))
    d_f3 = d_f3 + D3
    df3_share = np.zeros((area_num, 24))
    df3_share[2, :] = d_f3
    # 定义原始变量即功率
    p31 = np.zeros((1, 24))
    p32 = np.zeros((1, 24))
    p33 = np.zeros((1, 24))
    p34 = np.zeros((1, 24))
    p35 = np.zeros((1, 24))
    p3 = [p31, p32, p33, p34, p35]
    # 信息图矩阵
    Bij_3 = np.array([[0.2, 0.2, 0.6]])  # 需要共享的数据以及需要隐私保护
    Bij_share_3 = np.zeros((area_num, area_num))
    Bij_share_3[2, 0] = Bij_3[0, 0]
    Bij_share_3[2, 1] = Bij_3[0, 1]
    # 梯度更新
    LR = 0.0005 # 学习率 0.0002
    print("开始学习更新**************************")
    K = 100
    for k in range(K):


        f_sum = (Bij_share_1 + Bij_share_2 + Bij_share_3) @ (f1_share + f2_share + f3_share)

        # 更新区域1的参数
        f1 = f_sum[0, :].reshape(1, 24) + Bij_1[0, 0] * f1 + LR * d_f1
        d_1 = np.zeros((1, 24))
        for i in range(len(gen_num1)):
            p1[i] = np.minimum(P_max_1[i, :], np.maximum(P_min_1[i, :],
                                (f1 - coe_c_1[i, 1]) / (2 * coe_c_1[i, 0] + 2 * f1 * epison_1[i])))
            d_1 += (- p1[i] + epison_1[i] * p1[i] ** 2).astype('float64')
            print(p1[i])
        d_f1 = d_1 + D1

        # 更新区域2的参数
        f2 = f_sum[1, :].reshape(1, 24) + Bij_2[0, 1] * f2 + LR * d_f2
        d_2 = np.zeros((1, 24))
        for i in range(len(gen_num2)):
            p2[i] = np.minimum(P_max_2[i, :], np.maximum(P_min_2[i, :],
                                (f2 - coe_c_2[i, 1]) / (2 * coe_c_2[i, 0] + 2 * f2 * epison_2[i])))

            d_2 += (- p2[i] + epison_2[i] * p2[i] ** 2).astype('float64')
        d_f2 = d_2 + D2


        # 更新区域3的参数
        f3 = f_sum[2, :].reshape(1, 24) + Bij_3[0, 2] * f3 + LR * d_f3
        d_3 = np.zeros((1, 24))
        for i in range(len(gen_num3)):
            p3[i] = np.minimum(P_max_3[i, :], np.maximum(P_min_3[i, :],
                                                         (f3 - coe_c_3[i, 1]) / (2 * coe_c_3[i, 0] + 2 * f3 * epison_3[i])))
            d_3 += (- p3[i] + epison_3[i] * p3[i] ** 2).astype('float64')
        d_f3 = d_3 + D3


        f1_share[0, :] = f1
        f2_share[1, :] = f2
        f3_share[2, :] = f3

        print("*****************************")
        print("第k：{}".format(k + 1),"迭代")
        print("区域1为:{}".format(d_f1))
        print("区域2为:{}".format(d_f2))
        print("区域3为:{}".format(d_f3))
        # print("区域1的二范数为:{}".format(d_f1_l2))
        # print("区域2的二范数为:{}".format(d_f2_l2))
        # print("区域3的二范数为:{}".format(d_f3_l2))
        print("区域1的对偶变量:{}".format(f1))
        print("区域2的对偶变量:{}".format(f2))
        print("区域3的对偶变量:{}".format(f3))

    print("区域1的发电机1的出力结果：", p1[0])
    print("区域1的发电机2的出力结果：", p1[1])
    print("区域1的发电机3的出力结果：", p1[2])
    print("区域总出力结果：", p1[0] + p1[1] + p1[2] + p2[0] + p2[1] + p3[0] + p3[1] + p3[2] + p3[3] + p3[4])
        # print("区域2的发电机1的出力结果：", p2[0])
        # print("区域2的发电机2的出力结果：", p2[1])
        # print("区域3的发电机1的出力结果：", p3[0])
        # print("区域3的发电机2的出力结果：", p3[1])
        # print("区域3的发电机3的出力结果：", p3[2])
        # print("区域3的发电机4的出力结果：", p3[3])
        # print("区域3的发电机5的出力结果：", p3[4])

    print("结束学习**************************")

if __name__ == '__main__':
    ED_DGD_main()

