import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

def cec_random_number_seed(dim, lb, ub):
    x_max = ub #上界
    x_min = lb #下界
    size = 120#族群大小
    kai = size
    random_number = []
    for i in range(kai):
        pso = np.array([x_min + np.random.random()*(x_max - x_min) for i in range(dim)])# 粒子的位置
        random_number.append(pso)
    return random_number

#目標函数
def fit_fun(x):
    y = np.sum(x**2)
    return y

def IPPSO(FOBJ, lu, nfevalMAX, random_number, n_packs, n_coy, max_vel, W1, W2, K, P, C, C1, C2):

    # 最佳化問題變量
    D = lu.shape[1]
    VarMin = lu[0]
    VarMax = lu[1]

    # 主循環
    kai = 0
    nfeval = 0
    imm_kai = 0

    # 族群初始化（公式2）
    pop_total = n_packs * n_coy#總粒子數量為族群數*族群內粒子數目
    costs = np.zeros((1, pop_total))
    coyotes = np.array(random_number)
    g_best_costs = np.zeros((1, n_packs))
    g_best_coyotes = np.zeros((n_packs, D))
    vel = []

    rt_costs = np.zeros((1, pop_total)) # 粒子適應度
    rt_coyotes = np.array(random_number) # 粒子位置
    for i in range(pop_total):
        vl = [np.random.uniform(-max_vel, max_vel) for i in range(D)] # 單個粒子速度
        vel.append(vl)
    vel = np.array(vel)# 所有粒子的速度
    packs = np.random.permutation(pop_total).reshape(n_packs, n_coy)

    # 評估適應度（公式3）
    for c in range(pop_total):
        costs[0, c] = FOBJ(coyotes[c, :])

    p_costs = costs # 本粒子找到的最優解，自我認知部分
    p_coyotes = coyotes # 本粒子找到的最優解位置，自我認知部分

    nfeval = pop_total

    ibest = np.argsort(costs)#返回的是数组值从小到大的索引值
    rt_costs = costs[0, ibest]#排序適應度，社會經驗部分
    rt_coyotes = coyotes[ibest, :]#排序粒子位置，社會經驗部分
    # 輸出變量
    globalMin = np.min(costs[0, :])
    ibest = np.argmin(costs[0, :])
    globalParams = coyotes[ibest, :]
    fitness_val_list = []
    lastParams = []
    fes = []

    fitness_val_list.append(globalMin)
    fes.append(0)

    while nfeval < nfevalMAX:  # 停止標準
        # 評估適應
        for pop in range(pop_total):
            costs[0, pop] = FOBJ(coyotes[pop, :])
            # 擇優選擇新的自我認知與原自我認知的適應度大小，並保留最優解（公式14）
            if costs[0, pop] > p_costs[0, pop] and kai != 0 :
                p_costs[0, pop] = costs[0, pop]
                p_coyotes[pop, :] = coyotes[pop, :]
            elif kai == 0:
                p_costs[0, pop] = costs[0, pop]
                p_coyotes[pop, :] = coyotes[pop, :]
        ibest = np.argsort(costs)#返回的是数组值从小到大的索引值
        rt_costs = costs[0, ibest]#排序適應度，社會經驗部分
        rt_coyotes = coyotes[ibest, :]#排序粒子位置，社會經驗部分
        for rt_kai in range(len(rt_costs[0])):
            if globalMin > rt_costs[0, rt_kai]:
                globalMin = rt_costs[0, rt_kai]#取得costs的最小值   
                globalkai = rt_coyotes[0,rt_kai]#取得全局最佳位置
                break
        #約束最大fes
        if nfeval < nfevalMAX:
            nfeval += 120

        # 執行每個族群內的操作
        for p in range(n_packs):
            # 獲取每個族群的粒子
            coyotes_aux = coyotes[packs[p, :], :]
            costs_aux = costs[0, packs[p, :]]
            p_coyotes_aux = p_coyotes[packs[p, :], :]
            p_costs_aux = p_costs[0, packs[p, :]]
            vel_aux = vel[packs[p, :], :]

            # 找到當前群內的最好粒子（公式5）
            ind = np.argsort(costs_aux)
            costs_aux = costs_aux[ind]#排序適應度，社會經驗部分
            coyotes_aux = coyotes_aux[ind, :]#排序粒子位置，社會經驗部分
            p_costs_aux = p_costs_aux[ind]#排序適應度，自我認知部分
            p_coyotes_aux = p_coyotes_aux[ind, :]#排序粒子位置，自我認知部分
            vel_aux = vel_aux[ind, :] #排序速度
            c_alpha = coyotes_aux[0, :] #找出最佳粒子位址
            c_alpha_fes = p_costs_aux[0] #找出最佳粒子適應度
            # 更新g_best
            if g_best_costs[0, p] > c_alpha_fes and kai != 0:
                g_best_costs[0, p] = c_alpha_fes
                g_best_coyotes[p] = c_alpha
            elif kai == 0:
                g_best_costs[0, p] = c_alpha_fes
                g_best_coyotes[p] = c_alpha

            # 不同粒子族群
            new_coyotes = np.zeros((n_coy, D))
            new_value = vel_aux
            for c in range(n_coy):
                # 更新速度
                if p ==0:
                    w1 = W1 * np.exp((-nfeval**2) / (nfevalMAX**2)) # IPPSO慣性重量改進策略
                    new_value[c, :] = w1 * vel_aux[c, :] + C1 * np.random.random() * (p_coyotes_aux[c, :] - coyotes_aux[c, :]) \
                                + C2 * np.random.random() * (g_best_coyotes[p] - coyotes_aux[c, :])
                else:
                    w2 = W2 - np.sin((np.pi * nfeval) / (nfevalMAX * 2)) # IPPSO慣性重量改進策略
                    new_value[c, :] = w2 * vel_aux[c, :] + C1 * np.random.random() * (p_coyotes_aux[c, :] - coyotes_aux[c, :]) \
                                + C2 * np.random.random() * (g_best_coyotes[p] - coyotes_aux[c, :])
                # 如果粒子速度大於粒子最大速度則更新
                for max_v in range(D):
                    if new_value[c, max_v] > max_vel:
                        new_value[c, max_v] = max_vel
                    elif new_value[c, max_v] < -max_vel:
                        new_value[c, max_v] = -max_vel
                vel_aux[c, :] = new_value[c, :]
                # 更新位置
                new_coyotes[c, :] = coyotes_aux[c, :] + vel_aux[c, :]
                new_coyotes[c, :] = Limita(new_coyotes[c, :], D, VarMin, VarMax)
                # 對粒子群內所有的粒子進行更新，得到新的粒子個體）
                coyotes_aux[c, :] = new_coyotes[c, :]

            # 更新族群信息
            coyotes[packs[p], :] = coyotes_aux
            costs[0, packs[p]] = costs_aux
            vel[packs[p], :] = vel_aux
            # 找到當前群內的最好粒子（公式5）
            if imm_kai == P:
                ind = np.argsort(costs_aux)#返回的是数组值从小到大的索引值
                costs_aux = costs_aux[ind]#排序適應度，社會經驗部分
                coyotes_aux = coyotes_aux[ind, :]#排序粒子位置，社會經驗部分
                p_costs_aux = p_costs_aux[ind]#排序適應度，自我認知部分
                p_coyotes_aux = p_coyotes_aux[ind, :]#排序粒子位置，自我認知部分
                vel_aux = vel_aux[ind, :] #排序速度

            if imm_kai == P:
                # 族群之間移民、交換信息
                operator = int(np.around(n_coy/2))
                operator_totol = int(np.around(operator * C))
                if np.random.random() <= 0.5:
                    for i in range(operator_totol): 
                        offer = np.random.randint(operator) #族群傳送編號
                        receive = np.random.randint(operator) #族群接收編號
                        coyotes[packs[0,receive], :] = coyotes[packs[1,offer], :]
                        costs[0, packs[0, receive]] = costs[0, packs[1, offer]]
                        p_coyotes[packs[0,receive], :] = p_coyotes[packs[1,offer], :]
                        p_costs[0, packs[0, receive]] = p_costs[0, packs[1, offer]]
                        vel[packs[0,receive], :] = vel[packs[1,offer], :]
                else:
                    for i in range(operator_totol): 
                        offer = np.random.randint(operator) #族群傳送編號
                        receive = np.random.randint(operator) #族群接收編號
                        coyotes[packs[1, receive], :] = coyotes[packs[0,offer], :]
                        costs[0, packs[1, receive]] = costs[0, packs[0, offer]]
                        p_coyotes[packs[1, receive], :] = p_coyotes[packs[0,offer], :]
                        p_costs[0, packs[1, receive]] = p_costs[0, packs[0, offer]]
                        vel[packs[1,receive], :] = vel[packs[0,offer], :]
                if np.random.random() < 0.5:
                    for i in range(operator_totol): 
                        offer = np.random.randint(operator) + operator #族群傳送編號
                        receive = np.random.randint(operator) + operator #族群接收編號
                        coyotes[packs[0,receive], :] = coyotes[packs[1,offer], :]
                        costs[0, packs[0, receive]] = costs[0, packs[1, offer]]
                        p_coyotes[packs[0,receive], :] = p_coyotes[packs[1,offer], :]
                        p_costs[0, packs[0, receive]] = p_costs[0, packs[1, offer]]
                        vel[packs[0,receive], :] = vel[packs[1,offer], :]
                else:
                    for i in range(operator_totol): 
                        offer = np.random.randint(operator) + operator #族群傳送編號
                        receive = np.random.randint(operator) + operator #族群接收編號
                        coyotes[packs[1, receive], :] = coyotes[packs[0,offer], :]
                        costs[0, packs[1, receive]] = costs[0, packs[0, offer]]
                        p_coyotes[packs[1, receive], :] = p_coyotes[packs[0,offer], :]
                        p_costs[0, packs[1, receive]] = p_costs[0, packs[0, offer]]
                        vel[packs[1,receive], :] = vel[packs[0,offer], :]
                imm_kai = 1
        # 輸出變量（所有最佳粒子中最好的最佳粒子）
        globalMin = np.min(costs[0, :])
        ibest = np.argmin(costs)
        globalParams = coyotes[ibest, :]

        if kai == 0:
            fitness_val_list.append(globalMin)
            fes.append(nfeval)
            kai = kai + 1
        elif kai != 0 and fitness_val_list[-1] > globalMin:
            fitness_val_list.append(globalMin)
            fes.append(nfeval)
            kai = kai + 1
        elif kai != 0 and fitness_val_list[-1] <= globalMin:
            fitness_val_list.append(fitness_val_list[-1])
            fes.append(nfeval)
            kai = kai + 1
    if fes[-1] < nfevalMAX:
        fitness_val_list.append(fitness_val_list[-1])
        fes.append(nfevalMAX)

    return fitness_val_list, globalParams, fes

# 搜索空間約束
def Limita(X, D, VarMin, VarMax):
    for abc in range(D):
        X[abc] = max([min([X[abc], VarMax[abc]]), VarMin[abc]])

    return X



if __name__=="__main__":

    # 目標函數定義
    fobj = fit_fun# Function
    dim = 2 # 問題維度
    lu = np.zeros((2, dim)) # 邊界
    lu[0, :] = 0 # 下邊界
    lu[1, :] = 1000 # 上邊界

    # IPSO parameters
    n_packs = 2 # 族群
    n_pso = 60 # 族群內的粒子數量
    size = n_packs * n_pso # 粒子數量
    w1 = 0.9 # 粒子慣性權重
    w2 = 1 # 粒子慣性權重
    k = 5 # 兩個種群最先開始的獨立運算次數
    p = 5 # 兩個種群粒子移民後的獨立進化代數
    c = 0.05 # 兩個種群的移民機率
    C1=1.6
    C2=1.5
    max_vel = 1#粒子的最大速度
    nfevalmax = 6000       # 停止標準
    jikkenkai = 25#實驗次數

    fit_var_list_iter = []
    best_coa_iter = []
    fes_iter = []
    sd_list = []
    coa_len = 0

    for i in range(jikkenkai):
        random_number = cec_random_number_seed(dim, lu[0, 0], lu[1, 0])
        random_number=np.array(random_number)
        firtpar = [] # 紀錄初始位置
        firtpar.append(random_number[:, 0])
        firtpar.append(random_number[:, 1])
        # 將 COALC 應用於具有定義參數的問題
        #coa_gbest=全局最小值 par=全局參數 fes=適應函數
        coa_gbest, par, fes = IPPSO(fobj, lu, nfevalmax, random_number, n_packs, n_pso, max_vel,w1, w2, k, p, c, C1, C2)
       
        # Keep the global best
        fit_var_list_iter.append(coa_gbest)
        best_coa_iter.append(par)
        fes_iter.append(fes)
        sd_list.append(coa_gbest[-1])

    # -------------------判斷list大小不足--------------------
    fes_chigai = []#當前差值
    fes_chigai_iter = []#總實驗差值

    for i in range(len(fit_var_list_iter)):
        if coa_len < len(fit_var_list_iter[i]):
            coa_len = len(fit_var_list_iter[i])
        for ii in range(1,len(fit_var_list_iter[i])):
            if ii < (len(fit_var_list_iter[i])-1):
                fes_chigai.append(fes_iter[i][ii+1] - fes_iter[i][ii])
        fes_chigai_iter.append(fes_chigai)
        fes_chigai = []
    # -------------------補齊list大小不足的--------------------
    for i in range(len(fit_var_list_iter)):
        #若當次實驗fes數目不足則
        while coa_len > len(fit_var_list_iter[i]):
            for ii in range(len(fes_chigai_iter[i])):
                fesmax = np.max(fes_chigai_iter[i])
                if fes_chigai_iter[i][ii] == fesmax :
                    fit_var_list_iter[i].insert( ii+2, ((fit_var_list_iter[i][ii+1] + 
                        fit_var_list_iter[i][ii+2])/2))
                    fes_iter[i].insert( ii+2, ((fes_iter[i][ii+1] + fes_iter[i][ii+2])/2))
                    fes_chigai_iter[i].insert( ii+1, fes_chigai_iter[i][ii]/2)
                    fes_chigai_iter[i][ii] = fes_chigai_iter[i][ii]/2
                #若當次實驗fes數目足夠則跳脫
                if coa_len == len(fit_var_list_iter[i]):
                    break
    # -------------------總實驗取平均--------------------
    coa_fit_iter = np.zeros(len(fes_iter[0]))#仿製相同的list大小，補0
    for i in fes_iter:
        coa_fit_iter = np.array(coa_fit_iter) + np.array(i)
    coa_fit_iter = coa_fit_iter/jikkenkai

    fit_iter = np.zeros(len(fit_var_list_iter[0]))#仿製相同的list大小，補0
    for i in fit_var_list_iter:
        fit_iter = np.array(fit_iter) + np.array(i)
    fit_iter = fit_iter/jikkenkai
    standard_deviation = np.std(sd_list)
    print("IPPSO" + str(jikkenkai) + "次實驗平均值:" + str(fit_iter[-1]))
    print("IPPSO" + str(jikkenkai) + "次實驗標準差:" + str(standard_deviation))
    # -------------------檔案儲存--------------------
    data = pd.DataFrame({'Fitness Evaluations(FEs)':fes,'Fitness Value':fit_iter})
    ii = 0
    for i in fes_iter:
        data.insert(data.shape[1], 'Fitness Evaluations(FEs):' + str(ii), i)
        ii = ii + 1
    ii = 0
    for i in fit_var_list_iter:
        data.insert(data.shape[1], 'Fitness Value:' + str(ii), i)
        ii = ii + 1
    data.to_excel('ippso.xlsx',encoding='cp950')
    # # 粒子最佳位置
    # data = pd.DataFrame()#建立表格
    # ii = 0
    # for i in best_coa_iter:
    #     data.insert(data.shape[1], 'bestPRL_' + str(ii), i)
    #     ii = ii + 1
    # ii = 0
    # data.to_excel('ippso.xlsx',encoding='cp950')