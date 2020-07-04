import csv
import pandas as pd

import numpy as np
from scipy.optimize import fsolve, basinhopping
import random
import timeit
from matplotlib import pyplot as plt


# 根据解的精度确定染色体(chromosome)的长度
# 需要根据决策变量的上下边界来确定
def getEncodedLength(delta=0.0001, boundarylist=[]):
    # 每个变量的编码长度
    lengths = []
    for i in boundarylist:
        lower = i[0]
        upper = i[1]
        # lamnda 代表匿名函数f(x)=0,50代表搜索的初始解
        res = fsolve(lambda x: ((upper - lower) * 1 / delta) - 2 ** x + 1, 50)
        length = int(np.floor(res[0]))
        lengths.append(length)
    return lengths
    pass


# 随机生成初始编码种群
def getIntialPopulation(encodelength, populationSize):
    # 随机化初始种群为0
    chromosomes = np.zeros((populationSize, sum(encodelength)), dtype=np.uint8)
    for i in range(populationSize):
        chromosomes[i, :] = np.random.randint(0, 2, sum(encodelength))
        # print('chromosomes shape:', chromosomes.shape)
    return chromosomes


# 染色体解码得到表现型的解
def decodedChromosome(encodelength, chromosomes, boundarylist, delta=0.0001):
    """
    :param encodelength: 存储每个变量的编码长度
    :param chromosomes: 所有种群的染色体（populationSize， sum(encodelength)）
    :param boundarylist: 每个变量的边界
    :param delta: 变量变化的精度
    :return: 每个变量的解码
    """
    populations = chromosomes.shape[0]
    variables = len(encodelength)
    decodedvalues = np.zeros((populations, variables))
    for k, chromosome in enumerate(chromosomes):
        chromosome = chromosome.tolist()    # 转成列表list
        start = 0
        for index, length in enumerate(encodelength):
            # 将一个染色体进行拆分，得到染色体片段
            power = length - 1
            # 解码得到的10进制数字
            demical = 0
            for i in range(start, length - start):
                demical += chromosome[i] * (2 ** power)
                power -= 1
            lower = boundarylist[index][0]
            upper = boundarylist[index][1]
            decodedvalue = lower + demical * (upper - lower) / (2 ** length - 1)
            decodedvalues[k, index] = decodedvalue
            # 开始去下一段染色体的编码
            start = length
    return decodedvalues


# 得到个体的适应度值及每个个体被选择的累积概率
def getFitnessValue(func, chromosomesdecoded):
    # 得到种群规模和决策变量的个数
    population, nums = chromosomesdecoded.shape
    # 初始化种群的适应度值为0
    fitnessvalues = np.zeros((population, 1))
    # 计算适应度值
    for i in range(population):
        fitnessvalues[i, 0] = 1/func(chromosomesdecoded[i, :])  # 其余函数的策略
        # fitnessvalues[i, 0] = abs(func(chromosomesdecoded[i, :])) # 用于F04的策略
        # 计算每个染色体被选择的概率
    probability = fitnessvalues / np.sum(fitnessvalues)
    # 得到每个染色体被选中的累积概率
    cum_probability = np.cumsum(probability)
    return fitnessvalues, cum_probability


# 新种群选择
def selectNewPopulation(chromosomes, cum_probability):
    m, n = chromosomes.shape
    newpopulation = np.zeros((m, n), dtype=np.uint8)
    # 随机产生M个概率值
    randoms = np.random.rand(m)
    for i, randoma in enumerate(randoms):
        logical = cum_probability >= randoma
        index = np.where(logical == 1)
        # index是tuple,tuple中元素是ndarray
        newpopulation[i, :] = chromosomes[index[0][0], :]
    return newpopulation
    pass


# 新种群交叉
def crossover(population, fitnessValues, Pc=0.8):
    """
    :param population: 新种群
    :param Pc: 交叉概率默认是0.8
    :return: 交叉后得到的新种群
    """
    # 根据交叉概率计算需要进行交叉的个体个数
    m, n = population.shape
    # 初始化染色体的交叉概率
    Pcs = np.zeros((m, 1))
    randomPcs = np.random.rand(m)
    indexs = []
    # 求平均适应值
    averageFitnessValue = sum(fitnessValues)/m
    for i in range(m):
        if(fitnessValues[i]>averageFitnessValue) :
            Pcs[i] = Pc
        else:
            Pcs[i] = Pc * (max(fitnessValues) - fitnessValues[i])/(max(fitnessValues) - averageFitnessValue)
        if(randomPcs[i]<Pcs[i]):
            indexs.append(i)

    numbers = len(indexs)
    # 确保进行交叉的染色体个数是偶数个
    if numbers % 2 != 0:
        # numbers += 1
        indexs.append(np.random.randint(m))
    # 交叉后得到的新种群
    updatepopulation = np.zeros((m, n), dtype=np.uint8)
    # 产生随机索引
    # index = random.sample(range(m), numbers)  # sample(范围，个数)
    # 不进行交叉的染色体进行复制
    for i in range(m):
        if not indexs.__contains__(i):
            updatepopulation[i, :] = population[i, :]
    # crossover
    while len(indexs) > 0:
        a = indexs.pop()
        b = indexs.pop()
        # 随机产生一个交叉点
        crossoverPoint = random.sample(range(1, n), 1)
        crossoverPoint = crossoverPoint[0]
        # one-single-point crossover
        updatepopulation[a, 0:crossoverPoint] = population[a, 0:crossoverPoint]
        updatepopulation[a, crossoverPoint:] = population[b, crossoverPoint:]
        updatepopulation[b, 0:crossoverPoint] = population[b, 0:crossoverPoint]
        updatepopulation[b, crossoverPoint:] = population[a, crossoverPoint:]
    return updatepopulation
    pass


# 染色体变异
def mutation(population, Pm=0.01):
    """

    :param population: 经交叉后得到的种群
    :param Pm: 变异概率默认是0.01
    :return: 经变异操作后的新种群
    """
    updatepopulation = np.copy(population)
    m, n = population.shape
    # 计算需要变异的基因个数
    gene_num = np.uint8(m * n * Pm)
    # 将所有的基因按照序号进行10进制编码，则共有m*n个基因
    # 随机抽取gene_num个基因进行基本位变异
    mutationGeneIndex = random.sample(range(0, m * n), gene_num)
    # 确定每个将要变异的基因在整个染色体中的基因座(即基因的具体位置)
    for gene in mutationGeneIndex:
        # 确定变异基因位于第几个染色体（确定行）
        chromosomeIndex = gene // n
        # 确定变异基因位于当前染色体的第几个基因位（确定列）
        geneIndex = gene % n
        # mutation
        if updatepopulation[chromosomeIndex, geneIndex] == 0:
            updatepopulation[chromosomeIndex, geneIndex] = 1
        else:
            updatepopulation[chromosomeIndex, geneIndex] = 0
    return updatepopulation
    pass




def mult(x):
    """
    :param x: 输入列表
    :return: 列表中元素相乘的值
    """
    res = 1
    for i in range(len(x)):
        res *= x[i]
    return res
    pass
# 定义适应度函数    目标函数
def fitnessFunction():
    # return lambda x: sum([x[i] for i in range(30)])
    # return lambda x: sum([x[i] for i in range(30)]) + mult([x[i] for i in range(30)])
    # return lambda x: (sum([sum([x[j] for j in range(i)])**2 for i in range(30)]))
    # return lambda x: sum([x[i]**2 - 10*np.cos(2*np.pi*x[i]) + 10 for i in range(30)])
    return lambda x: 20*(1-np.exp(-0.2*np.sqrt(1/30*sum([x[i]**2 for i in range(30)])))) + np.exp(1) - np.exp(1/30*sum([np.cos(2*np.pi*x[i]) for i in range(30)]))
    # return lambda x: sum([-x[i]*np.sin(np.sqrt(x[i])) for i in range(30)])
    # return lambda x: 21.5 + x[0]*np.sin(4*np.pi*x[0]) + x[1]*np.cos(20*np.pi*x[1])

    pass

def getTrueValue(func, x):
    return func(x)
    pass

def main(max_iter=100):
    # 每次迭代得到的最优解
    optimalSolutions = []
    optimalValues = []
    F04optimalValues = []

    # 决策变量的取值范围
    # decisionVariables = [[-3.0, 12.1], [4.1, 5.8]]
    decisionVariables = np.zeros((30, 2))
    for i in range(30):
        # decisionVariables[i] = [0.00001, 100]
        # decisionVariables[i] = [0.00001, 10]
        # decisionVariables[i] = [0.00001, 500]
        # decisionVariables[i] = [0.00001, 5.12]
        decisionVariables[i] = [0.00001, 32]
    # 得到染色体编码长度
    lengthEncode = getEncodedLength(boundarylist=decisionVariables)
    # 得到初始种群编码
    chromosomesEncoded = getIntialPopulation(lengthEncode, 20)
    for iteration in range(max_iter):

        # 种群解码
        decoded = decodedChromosome(lengthEncode, chromosomesEncoded, decisionVariables)
        # 得到个体适应度值和个体的累积概率
        evalvalues, cum_proba = getFitnessValue(fitnessFunction(), decoded)
        # 选择新的种群
        newpopulations = selectNewPopulation(chromosomesEncoded, cum_proba)
        # 进行交叉操作
        crossoverpopulation = crossover(newpopulations, evalvalues)
        # mutation
        mutationpopulation = mutation(crossoverpopulation)

        # 将变异后的种群解码，得到每轮迭代最终的种群
        final_decoded = decodedChromosome(lengthEncode, mutationpopulation, decisionVariables)

        # 适应度评价
        fitnessvalues, cum_individual_proba = getFitnessValue(fitnessFunction(), final_decoded)
        chromosomesEncoded = selectNewPopulation(mutationpopulation, cum_individual_proba)

        # 搜索每次迭代的最优解，以及最优解对应的目标函数的取值
        optimalValues.append(np.max(list(fitnessvalues)))
        index = np.where(fitnessvalues == max(list(fitnessvalues)))
        optimalSolutions.append(final_decoded[index[0][0], :])
        # chromosomesEncoded[index[0][0], :] = mutationpopulation[index[0][0], :]
        F04optimalValues.append(getTrueValue(fitnessFunction(), final_decoded[index[0][0], :]))


        # 搜索最优解
    optimalValue = np.max(optimalValues)
    optimalIndex = np.where(optimalValues == optimalValue)
    optimalSolution = optimalSolutions[optimalIndex[0][0]]
    return optimalSolution, F04optimalValues, optimalValues


solution, F04value, values = main()
# print(solution[0], solution[1], solution[2])
print('最优解：', np.array(solution))
print('最优目标函数值:', '%.4f' % getTrueValue(fitnessFunction(), solution))
# 测量运行时间
elapsedtime = timeit.timeit(stmt=main, number=1)
print('Searching Time Elapsed:(S)', elapsedtime)
# 画图

# values = sorted(1/np.array(values), reverse=True)
values = 1/np.array(values)
# values = F04value
# plt.plot(values, color='r', linestyle='--')
# plt.show()

# 保存数据
dataframe = pd.DataFrame({'aGA': values})
# dataframe.to_csv(r"aGAresult.csv", sep=',', index=False)
# dataframe.to_csv(r"aGAresult-F02.csv", sep=',', index=False)
# dataframe.to_csv(r"aGAresult-F03.csv", sep=',', index=False)
# dataframe.to_csv(r"aGAresult-F04.csv", sep=',', index=False)
# dataframe.to_csv(r"aGAresult-F05.csv", sep=',', index=False)
dataframe.to_csv(r"aGAresult-F06.csv", sep=',', index=False)

