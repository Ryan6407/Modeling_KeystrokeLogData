import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm.notebook import tqdm
from scipy.spatial import distance
from sklearn.svm import SVR
import lightgbm
from statsmodels.tsa.ar_model import AutoReg
from IPython.display import display
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt 
import seaborn as sns
tqdm.pandas()

def get_lin_reg_coef(data):
    x_values = np.array([x for x in range(1, len(data)+1)])
    model = LinearRegression()
    model.fit(x_values.reshape(-1, 1), data)
    return model.coef_[0]

def get_ar_params(data):
    data = data.values

    model = AutoReg(data, lags=1)
    trained_model = model.fit()
    return trained_model.params

def get_shannon_entropy(data):
    entropy_data = [x/sum(data) for x in data]

    s_entropy = 0

    for p in entropy_data:
        if p > 0:
            s_entropy += p * math.log(p, 2)

    return -s_entropy

def get_shannon_jensen_div(data):
    data = np.array([x/sum(data) for x in data])
    uniform_dist = np.array([1/len(data) for x in data])
    return distance.jensenshannon(data, uniform_dist)

def get_extremes(data):
    data = data.values
    diffs = [data[i+1] - data[i] for i in range(len(data)-1)]
    extreme_count = 0

    for i in range(len(diffs)-1):
        if (diffs[i] < 0 and diffs[i+1] < 0) or (diffs[i] > 0 and diffs[i+1] > 0):
            pass
        else:
            extreme_count += 1

    return extreme_count

def get_avg_recurrence(data):
   # data = data.values

    total = 0
    count = 0
    last_non_0 = -1

    for idx in range(0, len(data)):
        if data[idx] != 0:
            total += idx - last_non_0 - 1
            count += 1
            last_non_0 = idx

    return total / count

def get_stddev_recurrence(data):
    data = data.values

    recurrences = []
    last_non_0 = -1

    for idx in range(0, len(data)):
        if data[idx] != 0:
            recurrences.append(idx - last_non_0 - 1)
            last_non_0 = idx

    return np.std(recurrences)

def get_cursor_back(data):
    return len([x for x in data.values if x < 0])

def count_bursts(data):
    data = data.values

    burst_count = 0

    burst_start = 0

    pause_total = 0

    pause_count = 0

    bursts = []

    for i in range(1, len(data)):
        if data[i] > 0.01:
            pause_total += data[i]
            pause_count += 1
            if i - burst_start - 1 > 0:
                burst_count += 1
                bursts.append(data[i])

            burst_start = i

    return [burst_count, pause_total/pause_count]

def feature_engineer(df):
    new_df = pd.DataFrame({"id" : list(df.groupby("id").groups.keys())})
    df_grouped = df.groupby("id")

    a = np.array(df_grouped.progress_apply(lambda x: count_bursts(x["diffs_seconds"])).values.tolist())

    new_df["mean_pause_duration"] = a[:,0]
    new_df["burst_count"] = a[:,1]

    new_df["verbosity"] = df_grouped.size().values
    backspace_df = df.groupby(["up_event", "id"]).size()["Backspace"]
    new_df = pd.merge(new_df, backspace_df.rename("backspaces"), on="id", how="left")

    new_df["word_count"] = df_grouped["word_count"].last().values

    period_df = df.groupby(["up_event", "id"]).size()["."]
    new_df = pd.merge(new_df, period_df.rename("sent_count"), on="id", how="left")

    enter_df = df.groupby(["up_event", "id"]).size()["Enter"]
    new_df = pd.merge(new_df, enter_df.rename("paragraph_count"), on="id", how="left")

    nonprod_df = df.groupby(["activity", "id"]).size()["Nonproduction"]
    new_df = pd.merge(new_df, nonprod_df.rename("Nonproduction"), on="id", how="left")

    new_df["avg_keystroke_speed"] = new_df["verbosity"] / df_grouped["time_elapsed"].tail(1).values

    ar_60 = np.array(df_grouped.progress_apply(lambda x: get_ar_params(x["window_60_sec_idx"].value_counts().reindex(range(max(x["window_60_sec_idx"])+1), fill_value=0))).values.tolist())
    ar_60_1 = ar_60[:,0]
    ar_60_2 = ar_60[:,1]
    new_df["ar_60_1"] = ar_60_1
    new_df["ar_60_2"] = ar_60_2

    ar_30 = np.array(df_grouped.progress_apply(lambda x: get_ar_params(x["window_30_sec_idx"].value_counts().reindex(range(max(x["window_30_sec_idx"])+1), fill_value=0))).values.tolist())
    ar_30_1 = ar_30[:,0]
    ar_30_2 = ar_30[:,1]
    new_df["ar_30_1"] = ar_30_1
    new_df["ar_30_2"] = ar_30_2

    new_df["largest_insert"] = df_grouped["word_diffs"].max().values
    new_df["largest_delete"] = df_grouped["word_diffs"].min().values

    new_df["backspaces"].fillna(0, inplace=True)
    new_df["largest_latency"] = df_grouped["diffs"].max().values
    new_df["smallest_latency"] = df_grouped["diffs"].min().values
    new_df["median_latency"] = df_grouped["diffs"].median().values
    new_df["first_pause"] = df.groupby("id").diffs_seconds.first().values
    new_df["pause_0.5"] = df[(df["diffs_seconds"] > 0.5) & (df["diffs_seconds"] < 1)].groupby("id").size().values
    new_df["pause_1"] = df[(df["diffs_seconds"] > 1) & (df["diffs_seconds"] < 1.5)].groupby("id").size().values
    new_df["pause_1.5"] = df[(df["diffs_seconds"] > 1.5) & (df["diffs_seconds"] < 2)].groupby("id").size().values
    pause_2_df = df[(df["diffs_seconds"] > 2) & (df["diffs_seconds"] < 3)].groupby("id").size()
    pause_3_df = df[df["diffs_seconds"] > 3].groupby("id").size()
    new_df = pd.merge(new_df, pause_2_df.rename("pause_2"), on="id", how="left")
    new_df = pd.merge(new_df, pause_3_df.rename("pause_3"), on="id", how="left")
    new_df["pause_2"].fillna(0, inplace=True)
    new_df["pause_3"].fillna(0, inplace=True)

    new_df["Slope_Degree_60"] = df_grouped.progress_apply(lambda x: get_lin_reg_coef(x["window_60_sec_idx"].value_counts().reindex(range(max(x["window_60_sec_idx"])+1), fill_value=0))).values
    new_df["Entropy_60"] = df_grouped.progress_apply(lambda x: get_shannon_entropy(x["window_60_sec_idx"].value_counts().reindex(range(max(x["window_60_sec_idx"])+1), fill_value=0))).values
    new_df["Degree_Uniformity_60"] = df_grouped.progress_apply(lambda x: get_shannon_jensen_div(x["window_60_sec_idx"].value_counts().reindex(range(max(x["window_60_sec_idx"])+1), fill_value=0))).values
    new_df["Local_Extremes_60"] = df_grouped.progress_apply(lambda x: get_extremes(x["window_60_sec_idx"].value_counts().reindex(range(max(x["window_60_sec_idx"])+1), fill_value=0))).values
    new_df["Average_Recurrence_60"] = df_grouped.progress_apply(lambda x: get_avg_recurrence(x["window_60_sec_idx"].value_counts().reindex(range(max(x["window_60_sec_idx"])+1), fill_value=0))).values
    new_df["StdDev_Recurrence_60"] = df_grouped.progress_apply(lambda x: get_stddev_recurrence(x["window_60_sec_idx"].value_counts().reindex(range(max(x["window_60_sec_idx"])+1), fill_value=0))).values

    new_df["Slope_Degree_30"] = df_grouped.progress_apply(lambda x: get_lin_reg_coef(x["window_30_sec_idx"].value_counts().reindex(range(max(x["window_30_sec_idx"])+1), fill_value=0))).values
    new_df["Entropy_30"] = df_grouped.progress_apply(lambda x: get_shannon_entropy(x["window_30_sec_idx"].value_counts().reindex(range(max(x["window_30_sec_idx"])+1), fill_value=0))).values
    new_df["Degree_Uniformity_30"] = df_grouped.progress_apply(lambda x: get_shannon_jensen_div(x["window_30_sec_idx"].value_counts().reindex(range(max(x["window_30_sec_idx"])+1), fill_value=0))).values
    new_df["Local_Extremes_30"] = df_grouped.progress_apply(lambda x: get_extremes(x["window_30_sec_idx"].value_counts().reindex(range(max(x["window_30_sec_idx"])+1), fill_value=0))).values
    new_df["Average_Recurrence_30"] = df_grouped.progress_apply(lambda x: get_avg_recurrence(x["window_30_sec_idx"].value_counts().reindex(range(max(x["window_30_sec_idx"])+1), fill_value=0))).values
    new_df["StdDev_Recurrence_30"] = df_grouped.progress_apply(lambda x: get_stddev_recurrence(x["window_30_sec_idx"].value_counts().reindex(range(max(x["window_30_sec_idx"])+1), fill_value=0))).values

    new_df["StDev_Events_60"] = df_grouped.apply(lambda x: x["window_60_sec_idx"].value_counts().reindex(x["window_60_sec_idx"].unique(), fill_value=0).std()).values
    new_df["StDev_Events_30"] = df_grouped.apply(lambda x: x["window_30_sec_idx"].value_counts().reindex(x["window_30_sec_idx"].unique(), fill_value=0).std()).values

    new_df["Cursor_Back_Count"] = df_grouped.progress_apply(lambda x: get_cursor_back(x["curpos_diffs"])).values
    new_df["Word_Back_Count"] = df_grouped.progress_apply(lambda x: get_cursor_back(x["word_diffs"])).values
   # new_df["prompt"] = df_grouped["Prompt"].first().map(cat2id).values

    return new_df


train_data = pd.read_csv("train_logs.csv")
train_targets = pd.read_csv("train_scores.csv")

train_data.sort_values(['id', 'up_time'], inplace=True)

train_data['diffs'] = train_data.groupby(['id'])['up_time'].transform(lambda x: x.diff())
#train_data["diffs"].fillna(0, inplace=True)

train_data.sort_index(inplace=True)
train_data["diffs_seconds"] = train_data["diffs"] / 1000

train_data["time_elapsed"] = train_data.groupby("id")["diffs_seconds"].cumsum()
train_data["time_elapsed"].fillna(0, inplace=True)

train_data['curpos_diffs'] = train_data.groupby(['id'])['cursor_position'].transform(lambda x: x.diff())
train_data['word_diffs'] = train_data.groupby(['id'])['word_count'].transform(lambda x: x.diff())

train_data["window_60_sec_idx"] = train_data["time_elapsed"].apply(lambda x: math.floor(x / 60.0))
train_data["window_30_sec_idx"] = train_data["time_elapsed"].apply(lambda x: math.floor(x / 30.0))

fe_train_data = feature_engineer(train_data)
full_df = pd.merge(fe_train_data, train_targets, on="id")
full_df.drop("id", axis=1, inplace=True)

from timeit import default_timer as timer

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

for fold, (train_idx, valid_idx) in enumerate(kf.split(full_df, full_df["score"]*2)):

    training_data, validation_data = full_df.iloc[train_idx], full_df.iloc[valid_idx]

    training_targets, validation_targets = training_data.pop("score"), validation_data.pop("score")

    start = timer()

    model = LGBMRegressor(verbose=0, n_estimators=10000, learning_rate=0.1)

    model.fit(training_data, training_targets, eval_set=(validation_data, validation_targets), early_stopping_rounds=100, verbose=0)
    valid_pred = model.predict(validation_data)

    end = timer()

    loss = mean_squared_error(validation_targets, valid_pred, squared=False)
    print(loss)
    print(end - start)

import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
lightgbm.plot_importance(model, figsize=(25, 20))