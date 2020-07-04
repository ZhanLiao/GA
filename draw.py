"""
@author: zhanliao
@what: 遗传算法的基本操作函数
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


# df1 = pd.read_csv('GAresult.csv')
df1 = pd.read_csv('GAresult-F02.csv')
# df1 = pd.read_csv('GAresult-F03.csv')
# df1 = pd.read_csv('GAresult-F04.csv')
# df1 = pd.read_csv('GAresult-F05.csv')
# df1 = pd.read_csv('GAresult-F06.csv')
data1 = df1.values.tolist()

# df2 = pd.read_csv('aGAresult.csv')
df2 = pd.read_csv('aGAresult-F02.csv')
# df2 = pd.read_csv('aGAresult-F03.csv')
# df2 = pd.read_csv('aGAresult-F04.csv')
# df2 = pd.read_csv('aGAresult-F05.csv')
# df2 = pd.read_csv('aGAresult-F06.csv')
data2 = df2.values.tolist()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文问题
plt.rcParams['axes.unicode_minus'] = False    # 显示负数问题
plt.plot(list(range(1, 101)), data1, label='GA', color='r', linewidth=2, linestyle=':') # 添加linestyle设置线条类型
plt.plot(list(range(1, 101)), data2, label='advancedGA', color='b', linewidth=2, linestyle='--')
plt.legend(loc='upper right')
plt.ylabel("函数值", fontsize=12)      # 设置纵轴单位
plt.xlabel("迭代次数", fontsize=12)  # 设置横轴单位
# plt.title("GA & advancedGA 关于函数F01的收敛对比曲线", fontsize=14)    # 设置图片的头部
plt.title("GA & advancedGA 关于函数F02的收敛对比曲线", fontsize=14)    # 设置图片的头部
# plt.title("GA & advancedGA 关于函数F03的收敛对比曲线", fontsize=14)    # 设置图片的头部
# plt.title("GA & advancedGA 关于函数F04的收敛对比曲线", fontsize=14)    # 设置图片的头部
# plt.title("GA & advancedGA 关于函数F05的收敛对比曲线", fontsize=14)    # 设置图片的头部
# plt.title("GA & advancedGA 关于函数F06的收敛对比曲线", fontsize=14)    # 设置图片的头部
plt.grid(True, linestyle="--", color='gray', linewidth='0.5', axis='both')
plt.show()