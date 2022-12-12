# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:36:15 2022

@author: tbednall
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

train_set = pd.read_csv("train_set.csv")
X_train = train_set[["fiscal_year_c", "business_c", "business_area_c", "career_level_adj_c", "hire_type_c", "tenure_bins_c", "til_bins_c", "othercomp_quartile_n", "pto_hrs_per_ww_bins_c", "low_perf_pct_adj_n", "next_level_pct_adj_n", "rpm_avg_adj_n", "rpm_tl_bins_adj_c", "sal_quartile_n", "exp_recent_hire_c", "util_flag_c", "mgd_rev_bins_c", "sales_bins_c"]]
Y_train = train_set[["vol_term_c"]]

test_set = pd.read_csv("test_set.csv")
X_test = test_set[["fiscal_year_c", "business_c", "business_area_c", "career_level_adj_c", "hire_type_c", "tenure_bins_c", "til_bins_c", "othercomp_quartile_n", "pto_hrs_per_ww_bins_c", "low_perf_pct_adj_n", "next_level_pct_adj_n", "rpm_avg_adj_n", "rpm_tl_bins_adj_c", "sal_quartile_n", "exp_recent_hire_c", "util_flag_c", "mgd_rev_bins_c", "sales_bins_c"]]
Y_test = test_set[["vol_term_c"]]

all_data = pd.concat([X_train, X_test], axis = 0, keys=("train","test"))

# Convert to categorical
for col in all_data.columns:
    if all_data[col].dtype=="object":
        df2 = pd.get_dummies(all_data[col], drop_first=True)       
        all_data = all_data.drop(columns=[col])
        all_data = pd.concat([all_data, df2], axis=1, join="outer")

X_train = all_data.loc[all_data.index.get_level_values(0)=="train",:]
X_test = all_data.loc[all_data.index.get_level_values(0)=="test",:]

rnd_clf = RandomForestClassifier(n_estimators = 100, max_depth = 20)
rnd_clf.fit(X_train, Y_train.loc[:, "vol_term_c"])

Y_train["predict"] = rnd_clf.predict(X_train).tolist()
Y_test["predict"] = rnd_clf.predict(X_test).tolist()

confusion_matrix(Y_train["vol_term_c"], Y_train["predict"])
confusion_matrix(Y_test["vol_term_c"], Y_test["predict"])

# How to interpret Confusion Matrix:
# True Negative  | False Positive
# False Negative | True Positive