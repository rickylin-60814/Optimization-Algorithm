import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

def cec_random_number_seed(dim, lb, ub):
    x_max = ub #上界
    x_min = lb #下界
    size = 60#族群大小
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

def COALC(FOBJ, lu, nfevalMAX, random_number, n_packs, n_coy):

    # 最佳化問題變量
    D = lu.shape[1]
    VarMin = lu[0]
    VarMax = lu[1]

    # 主循環
    kai = 0
    year = 1
    nfeval = 0

    #最低的郊狼數目
    if n_coy < 3:
        raise Exception("At least 3 coyotes per pack must be used")

    # 離群概率
    p_leave = 0.005*(n_coy**2)
    Ps = 1/D

    # 族群初始化（公式2）
    pop_total = n_packs*n_coy#總郊狼數量
    costs = np.zeros((1, pop_total))
    coyotes = np.array(random_number)
    ages = np.zeros((1, pop_total))
    packs = np.random.permutation(pop_total).reshape(n_packs, n_coy)

    # 評估適應度（公式3）
    for c in range(pop_total):
        costs[0, c] = FOBJ(coyotes[c, :])

    nfeval = pop_total

    # 輸出變量
    globalMin = np.min(costs[0, :])
    ibest = np.argmin(costs[0, :])
    globalParams = coyotes[ibest, :]
    fitness_val_list = []
    lastParams = []
    fes = []

    fitness_val_list.append(globalMin)
    fes.append(0)
    # 主循環
    year = 1
    while nfeval < nfevalMAX:  # 停止標準
        # 更新郊狼年齡計數器
        year += 1

        # 執行每個族群內的操作
        for p in range(n_packs):
            # 獲取每個族群的郊狼
            coyotes_aux = coyotes[packs[p, :], :]
            costs_aux = costs[0, packs[p, :]]
            ages_aux = ages[0, packs[p, :]]

            # 找到當前群內的頭狼（公式5）
            ind = np.argsort(costs_aux)
            costs_aux = costs_aux[ind]
            coyotes_aux = coyotes_aux[ind, :]
            ages_aux = ages_aux[ind]
            c_alpha = coyotes_aux[0, :]

            # 計算群體的社會趨勢（公式6）
            tendency = np.median(coyotes_aux, 0)

            #  更新郊狼的社會狀況
            new_coyotes = np.zeros((n_coy, D))
            for c in range(n_coy):
                rc1 = c
                while rc1 == c:
                    rc1 = np.random.randint(n_coy)
                rc2 = c
                while rc2 == c or rc2 == rc1:
                    rc2 = np.random.randint(n_coy)

                # 利用線性控制參數控制
                a = 0.8 - nfeval * ((0.4) / nfevalMAX) # 利用線性控制參數控制
                r1 = np.random.rand()
                r2 = np.random.rand()
                a1 = a * r1 * 2
                a2 = a * np.random.uniform(0,a1) * 2

                # 嘗試根據社會狀況更新
                # 計算頭狼與群體趨勢對當前時刻對應的郊狼群內個體更新產生的影響δ1和δ2
                new_coyotes[c, :] = coyotes_aux[c, :] + a1*(c_alpha - coyotes_aux[rc1, :]) + \
                                    a2*(tendency - coyotes_aux[rc2, :])#ccoa

                new_coyotes[c, :] = Limita(new_coyotes[c, :], D, VarMin, VarMax)

                # 郊狼進行更新（公式13）
                new_cost = FOBJ(new_coyotes[c, :])
                nfeval += 1

                # 根據貪心演算法更新位置（公式14）
                if new_cost < costs_aux[c]:
                    costs_aux[c] = new_cost
                    coyotes_aux[c, :] = new_coyotes[c, :]

            hunter = np.around(n_coy*(a*0.1))#np.around四捨五入 np.ceil無條件進位 被獵人殺死的機率為5%
            for kill in range(1,int(hunter+1)):
                #新狐狸以游牧狐狸的身份離開棲息地，並走出該地區尋找食物以及在他們的群體中繁殖的可能性
                k = np.random.random()#隨機參數k定義替換
                habitat_center = (coyotes_aux[0, :]+coyotes_aux[1, :])/2
                ages_aux[n_coy-kill] = 0
                coyotes_aux[n_coy-kill, :] = k * habitat_center
                # 將郊狼保持在搜索空間中（優化問題約束）
                coyotes_aux[n_coy-kill, :] = Limita(coyotes_aux[n_coy-kill, :], D, VarMin, VarMax)

            # 來自隨機郊狼父母誕生（公式7和公式1）
            parents = np.random.permutation(n_coy)[:2]
            prob1 = (1-Ps)/2
            prob2 = prob1
            pdr = np.random.permutation(D)
            p1 = np.zeros((1, D))
            p2 = np.zeros((1, D))
            p1[0, pdr[0]] = 1
            p2[0, pdr[1]] = 1
            r = np.random.rand(1, D-2)
            p1[0, pdr[2:]] = r < prob1
            p2[0, pdr[2:]] = r > 1-prob2

            n = np.logical_not(np.logical_or(p1, p2))

            # 考慮內在和外在影響生成幼崽
            pup = p1*coyotes_aux[parents[0], :] + \
                  p2*coyotes_aux[parents[1], :] + \
                  n*(VarMin + np.random.rand(1, D) * (VarMax - VarMin))

            # 驗證幼狼是否會存活
            pup_cost = FOBJ(pup[0, :])
            nfeval += 1
            worst = np.flatnonzero(costs_aux > pup_cost)
            if len(worst) > 0:
                older = np.argsort(ages_aux[worst])
                which = worst[older[::-1]]
                coyotes_aux[which[0], :] = pup
                costs_aux[which[0]] = pup_cost
                ages_aux[which[0]] = 0

            # 更新族群信息
            coyotes[packs[p], :] = coyotes_aux
            costs[0, packs[p]] = costs_aux
            ages[0, packs[p]] = ages_aux

        # 郊狼離群（公式4）
        if n_packs > 1:
            if np.random.rand() < p_leave:
                rp = np.random.permutation(n_packs)[:2]
                rc = [np.random.randint(0, n_coy), np.random.randint(0, n_coy)]
                aux = packs[rp[0], rc[0]]
                packs[rp[0], rc[0]] = packs[rp[1], rc[1]]
                packs[rp[1], rc[1]] = aux

        # 更新郊狼年齡
        ages += 1

        # 輸出變量（所有頭狼中最好的頭狼）
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
    lu[0, :] = -1000 # 下邊界
    lu[1, :] = 1000 # 上邊界

    # COA parameters
    n_packs = 10            # 族群
    n_coy = 6               # 郊狼數量
    size = n_packs * n_coy  # 郊狼數量
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
        coa_gbest, par, fes = COALC(fobj, lu, nfevalmax, random_number, n_packs, n_coy)
       
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
    print("COALC" + str(jikkenkai) + "次實驗平均值:" + str(fit_iter[-1]))
    print("COALC" + str(jikkenkai) + "次實驗標準差:" + str(standard_deviation))
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
    data.to_excel('coalc.xlsx',encoding='cp950')
    # # 粒子最佳位置
    # data = pd.DataFrame()#建立表格
    # ii = 0
    # for i in best_coa_iter:
    #     data.insert(data.shape[1], 'bestPRL_' + str(ii), i)
    #     ii = ii + 1
    # ii = 0
    # data.to_excel('coalc.xlsx',encoding='cp950')