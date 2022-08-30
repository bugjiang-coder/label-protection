import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is ", device)
raw_df = pd.read_csv(
    "creditcard.csv"
)
# 将读入的数据分成两类
raw_df_neg = raw_df[raw_df["Class"] == 0]
raw_df_pos = raw_df[raw_df["Class"] == 1]

down_df_neg = raw_df_neg  # .sample(40000)
# 基于同一轴将多个数据集合并  为啥要拆开再合并呢？
down_df = pd.concat([down_df_neg, raw_df_pos])

neg, pos = np.bincount(down_df["Class"])
total = neg + pos
print(
    "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
        total, pos, 100 * pos / total
    )
)

