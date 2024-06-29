import numpy as np
import time
import matplotlib.pyplot as plt
import random
from pypower.api import runopf, ppoption, runpf, case39

# 以简单的经济调度为例
def data_process(ar):
    Data = case39()  # 读取IEEE30节点数据
    area = ar
    area_num = 3
    node = Data['bus']  # 读取节点数据
    T = int(24)
    bus_num = []
    for i in range(node.shape[0]):
        if node[i, 6] == area:
            bus_num.append(node[i, 0].astype('int'))  # 节点编号
    # print(bus_num) 区域1的节点编号[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 31, 32, 39]
    if ar == 1:
        demand = np.array([116.59120834, 123.98688577, 113.42161175, 146.9268676,  146.90878,
  116.48199214, 102.80749221, 108.64197309, 148.38283856, 122.00368536,
  141.92794801, 119.3697753,  118.84842478, 142.4101116,  124.88921629,
  144.98886753, 116.62752822, 137.16369564, 127.03230429, 112.5336854,
  143.9660644,  140.55540449, 110.8201987,  123.46946901]).astype('float64')
    if ar == 2:
        demand = np.array([117.87464739, 145.58995372, 128.80375944, 104.75418213, 106.59168313,
  117.37732454, 114.80746794, 144.15697098, 122.55366449, 145.00127588,
  143.34442045, 120.38135411, 137.62103673, 112.93036212, 125.65651644,
  119.20684215, 143.66254236, 133.46705408, 125.02064271, 131.31111562,
  144.81547165, 114.05355178, 109.99479943, 141.87938777]).astype('float64')
    if ar == 3:
        demand = np.array([127.64380315, 146.01824739, 133.39303795, 141.53531162, 102.86401083,
  140.71977157, 103.63988829, 146.37819889, 117.77498113, 142.60051059,
  142.92005007, 145.85118633, 100.06519378, 128.39341705, 132.31439301,
  137.92290536, 119.18205036, 108.76907669, 136.85500847, 127.05589272,
  107.94607255, 127.8648487,  126.59407505, 142.430998]).astype('float64')

    G_Data = Data['gen']  # 读取发电机节点数据
    gen_num = []
    for i in range(G_Data.shape[0]):
        if G_Data[i, 0] in bus_num:
            gen_num.append(G_Data[i, 0].astype('int'))  # 节点编号
    # print(gen_num) [31, 32, 39][1, 2, 9]
    genP_max = [] # 发电机有功功率上限
    genP_min = [] # 发电机有功功率下限
    for i in gen_num:
        genP_max.append(G_Data[i - 30, 8])
        genP_min.append(G_Data[i - 30, 9])

    P_max = np.zeros((len(gen_num), T))
    for i in range(len(P_max)):
        for j in range(T):
            P_max[i][j] = genP_max[i]

    P_min = np.zeros((len(gen_num), T))
    for i in range(len(P_min)):
        for j in range(T):
            P_min[i][j] = genP_min[i]

    RU = np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30])  # 爬坡约束
    RD = np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30])
    if ar == 1:
        epison = np.array([0.00003526, 0.00003526, 0.00005421]).astype('float64')
        # RU = np.array([50, 50, 50])  # 爬坡约束
        # RD = np.array([50, 50, 50])
    if ar == 2:
        epison = np.array([0.00003471, 0.00003471]).astype('float64')
    if ar == 3:
        epison = np.array([0.00002421, 0.00002421, 0.00002421, 0.00003143, 0.00003143]).astype('float64')
    gen_cost = Data['gencost']
    coe_c = np.zeros((len(gen_num), 3))
    for i in range(len(gen_num)):
        for j in range(3):
            coe_c[i, :] = gen_cost[gen_num[i] - 30, 4:7]
    return demand, P_max, P_min, coe_c, epison, gen_num



