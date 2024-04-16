#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import time
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from scipy.stats import norm
import GPy

# In[ ]:

# Run this block only if the modules below are not installed
os.system('pip install GPy')
os.system('pip install scikit-learn')
os.system('pip install scipy')

# In[ ]:

# Parameter setting
BOA = 1  # 1 or 2 (1: early optimization stage, 2: late optimization stage)
number_of_experiments_per_cycle = 5

# Label in DoE_Exp_Table_IVT_1FoM.xlsx
output_label = "mRNA-yield"

# In[ ]:

xi = 0.01
reject_rad = 1.
core_opt = False

if os.path.exists("./GP"):
    num = 1
    while True:
        if not os.path.exists(f"./GP_prev{num}"):
            os.rename("./GP", f"./GP_prev{num}")
            break
        else:
            num += 1
os.makedirs("./GP", exist_ok=True)

exp_table = pd.read_excel("./DoE_Exp_Table_IVT_1FoM.xlsx", index_col=0)
exp_table.columns = [c.strip() for c in exp_table.columns]

x_data_column = [c for c in exp_table.columns if not c == output_label]
print(f"[Data read] factors: {x_data_column}")

reso = 11
if len(x_data_column) > 6:
    print("[CAUTION] The number of the factors is large. The search grid resolution is lowered.")
    reso -= 2 * (len(x_data_column) - 6)
    print(f"Resolution={reso} (default:11)")
if reso < 5:
    print("[ERROR] The grid resolution is too low. Use sciCORE instead of this laptop.")
    print("System terminated.")
else:

    min_li = [exp_table.loc["MIN", c] for c in x_data_column]
    max_li = [exp_table.loc["MAX", c] for c in x_data_column]
    min_max_li = np.array([min_li, max_li], dtype=float)

    mmscaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    mmscaler.fit(min_max_li)

    exp_table = exp_table.drop(["MIN", "MAX"])
    original_size = len(exp_table)
    start = time.time()

    if BOA == 1:
        for i in range(1, number_of_experiments_per_cycle + 1):
            print(f"[Cycle {i}] {time.time() - start:.2f}[sec]")
            # print(exp_table)
            x_train = mmscaler.transform(exp_table.loc[:, x_data_column].values)
            y_train = exp_table.loc[:, [output_label]].values

            kern = GPy.kern.RBF(len(x_data_column), ARD=True)
            gpy_model = GPy.models.GPRegression(X=x_train, Y=y_train, kernel=kern, normalizer=True)
            if core_opt:
                gpy_model.optimize(messages=True, max_iters=100000)

            lis = []
            for j in range(len(x_data_column)):
                lis += [np.linspace(0, 1.0, reso)]
            points = np.array(list(itertools.product(*lis)))

            minDist = distance.cdist(points, x_train, metric='euclidean').min(axis=1)
            points = points[minDist > 0.01]

            if i > 1:
                x_train_tentative = x_train[-(i - 1):, :]
                # print(x_train_tentative)
                minDist = distance.cdist(points, x_train_tentative, metric='euclidean').min(axis=1)
                points = points[minDist > reject_rad]

            GO_table = pd.DataFrame(points, columns=[f"{c}_S" for c in x_data_column])

            pred_mean, pred_var = gpy_model.predict(points)
            pred_mean = pred_mean.reshape(-1)
            pred_std = np.sqrt(pred_var.reshape(-1))
            GO_table["pred_mean"] = pred_mean
            GO_table["pred_std"] = pred_std

            mu_sample, _ = gpy_model.predict(x_train)
            mu_sample_opt = np.max(mu_sample)

            with np.errstate(divide='warn'):
                imp = pred_mean - mu_sample_opt - xi
                Z = imp / pred_std
                ei = imp * norm.cdf(Z) + pred_std * norm.pdf(Z)
                ei[pred_std == 0.] = 0.

            GO_table["Acquisition"] = ei

            for c in x_data_column:
                GO_table[c] = 0.

            GO_table.loc[:, x_data_column] = mmscaler.inverse_transform(points)

            GO_table = GO_table.sort_values("Acquisition", ascending=False)
            GO_table[:1000].to_csv(f"./GP/GP_{i}.csv")

            next_index = len(exp_table) + 1
            exp_table.loc[next_index] = -1
            top_data = GO_table.iloc[0]
            for clm in x_data_column:
                exp_table.loc[next_index, clm] = top_data[clm]
            exp_table.loc[next_index, output_label] = top_data["pred_mean"]

    elif BOA == 2:
        for i in range(1, number_of_experiments_per_cycle + 1):
            print(f"[Cycle {i}] {time.time() - start:.2f}[sec]")
            x_train = mmscaler.transform(exp_table.loc[:, x_data_column].values)
            y_train_raw = exp_table.loc[:, [output_label]].values

            # scaling y
            y_scale_val = y_train_raw.max() / 10.
            y_train = y_train_raw / y_scale_val

            kern = GPy.kern.RBF(len(x_data_column), ARD=True)
            gpy_model = GPy.models.GPRegression(X=x_train, Y=y_train, kernel=kern, normalizer=True)
            if core_opt:
                gpy_model.optimize(messages=True, max_iters=100000)

            lis = []
            for j in range(len(x_data_column)):
                lis += [np.linspace(0, 1.0, reso)]
            points = np.array(list(itertools.product(*lis)))

            minDist = distance.cdist(points, x_train, metric='euclidean').min(axis=1)
            points = points[minDist > 0.01]

            GO_table = pd.DataFrame(points, columns=[f"{c}_S" for c in x_data_column])

            pred_mean, pred_var = gpy_model.predict(points)
            pred_mean = pred_mean.reshape(-1)
            pred_std = pred_var.reshape(-1)
            GO_table["pred_mean_real"] = pred_mean * y_scale_val
            GO_table["pred_mean_scaled"] = pred_mean
            GO_table["pred_std"] = pred_std

            mu_sample, _ = gpy_model.predict(x_train)
            mu_sample_opt = np.max(mu_sample)

            with np.errstate(divide='warn'):
                imp = pred_mean - mu_sample_opt - xi
                Z = imp / pred_std
                ei = imp * norm.cdf(Z) + pred_std * norm.pdf(Z)
                ei[pred_std == 0.] = 0.

            GO_table["Acquisition"] = ei

            for c in x_data_column:
                GO_table[c] = 0.

            GO_table.loc[:, x_data_column] = mmscaler.inverse_transform(points)

            GO_table = GO_table.sort_values("Acquisition", ascending=False)
            GO_table[:1000].to_csv(f"./GP/GP_{i}.csv")

            next_index = len(exp_table) + 1
            exp_table.loc[next_index] = -1
            top_data = GO_table.iloc[0]
            for clm in x_data_column:
                exp_table.loc[next_index, clm] = top_data[clm]
            exp_table.loc[next_index, output_label] = top_data["pred_mean_real"]

    else:
        print("[ERROR] The setting 'BOA' must be 1 or 2.")

    for i in range(len(exp_table) + 1):
        if i > original_size:
            exp_table.loc[i, output_label] = -1
    exp_table.to_excel("./GP/DoE_Result.xlsx")
    print(f"[Done] {time.time() - start:.2f}[sec]")
