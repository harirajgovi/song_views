# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:00:50 2020

@author: hari
"""

#importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import re

#importing datas
df_train = pd.read_csv("Data_Train.csv")
df_test = pd.read_csv("Data_Test.csv")

#cleaning datas
for df in [df_train, df_test]:
    for col in ["Likes", "Popularity"]:
        df[col].replace(to_replace=r"[,]", value="", inplace=True, regex=True)
        err_str = str(list(df[col]))
        err_val = re.findall(r"\d+[.]{1}\d+[K]", err_str)
        val = [s[:s.index(".")]+s[s.index(".")+1:-1]+"00" for s in err_val]
        df[col].replace(to_replace=err_val, value=val, inplace=True)
        err_val = re.findall(r"\d+[.]{1}\d+[M]", err_str)
        val = [s[:s.index(".")]+s[s.index(".")+1:-1]+"00000" for s in err_val]
        df[col].replace(to_replace=err_val, value=val, inplace=True)
        df[col].replace(to_replace=r"[K]", value="000", inplace=True, regex=True)
        df[col].replace(to_replace=r"[M]", value="000000", inplace=True, regex=True)
 
for df in [df_train, df_test]:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Timestamp"] = df["Timestamp"].apply(lambda x: x.year)
    df.rename(columns={"Timestamp":"Year"}, inplace=True)
   
uniq_id = df_test[["Unique_ID"]]
for df in [df_train, df_test]:
    df.drop(columns=["Unique_ID", "Country", "Song_Name"], axis=1, inplace=True)

cols = ["Year", "Views", "Comments", "Likes", "Popularity", "Followers"]
df_train[cols] = df_train[cols].astype(int)
cols = ["Year", "Comments", "Likes", "Popularity", "Followers"]
df_test[cols] = df_test[cols].astype(int)
df_train[["Name", "Genre"]] = df_train[["Name", "Genre"]].astype(str)
df_test[["Name", "Genre"]] = df_test[["Name", "Genre"]].astype(str)

#splitting datas into categorical and numerical 
df_train_catgry = df_train[[col for col in df_train.columns if df_train[col].dtype == object]]
df_train_numeric = df_train[[col for col in df_train.columns if df_train[col].dtype != object]]
df_test_catgry = df_test[[col for col in df_test.columns if df_test[col].dtype == object]]
df_test_numeric = df_test[[col for col in df_test.columns if df_test[col].dtype != object]]

# treating categorical datas 
df_train_catgry.replace("󠀠", np.nan, inplace=True)
df_test_catgry.replace("󠀠", np.nan, inplace=True)
for f in df_train_catgry.columns:
    mode = df_train_catgry[f].mode()[0]
    df_train_catgry[f].fillna(mode, inplace=True)
    df_test_catgry[f].fillna(mode, inplace=True)
df_train_catgry = df_train_catgry[["Name", "Genre"]]
combined_name = pd.concat([df_train_catgry, df_test_catgry])
combined_name = pd.get_dummies(combined_name, drop_first=True)
df_train_catgry = combined_name.head(df_train_catgry.shape[0])
df_test_catgry = combined_name.tail(df_test_catgry.shape[0])

# treating numerical datas
for f in df_train_numeric.columns:
    median = df_train_numeric[f].median()
    df_train_numeric[f].fillna(median, inplace=True)
    if f != "Views":
        df_test_numeric[f].fillna(median, inplace=True)

# combining categorical and numerical datas
df_train = pd.concat([df_train_catgry, df_train_numeric], axis=1)
df_test = pd.concat([df_test_catgry, df_test_numeric], axis=1)
target = df_train[["Views"]]
df_train.drop(columns=["Views"], axis=1, inplace=True)

# Standardization 
std_scl = StandardScaler()
df_train_scl = pd.DataFrame(std_scl.fit_transform(df_train), columns=list(df_train.columns))
df_test_scl = pd.DataFrame(std_scl.transform(df_test), columns=list(df_test.columns))
scl_y = StandardScaler()
target_scl = pd.DataFrame(scl_y.fit_transform(target), columns=list(target.columns))

# converting data into arrays
x_train = df_train_scl.iloc[:, :].values
y_train = target_scl.iloc[:, 0].values
x_test = df_test_scl.iloc[:, :].values

# ML model
regressor = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=6000, min_child_weight=0, 
                         subsample=0.9, colsample_bytree=0.9, gamma=0.0001, reg_alpha=0,
                         objective="reg:squarederror", n_jobs=-1, random_state=0)

#hyper-parameter tuning using gridsearch
params = {"learning_rate": [0.1, 0.09, 0.2], "n_estimators": [100, 200, 300]}
tuner = GridSearchCV(regressor, param_grid=params, scoring="neg_mean_squared_error", 
                     n_jobs=-1, cv=5)
tuned = tuner.fit(x_train, y_train)
best_params = tuned.best_params_
best_score = tuned.best_score_

#Training model
print("\nTraining XGBR model.....")
regressor.fit(x_train, y_train)
imp_fetur = pd.Series(regressor.feature_importances_,
                index=list(range(0, 1242))).sort_values(ascending=False)

#predicting target
train_pred = regressor.predict(x_train)
test_pred = regressor.predict(x_test)

#estimation
estimators = {"abs_error":"neg_mean_absolute_error", 
              "squared_error":"neg_mean_squared_error", 
              "r2_score":"r2"}
cross_val = cross_validate(regressor, x_train, y_train, 
                           scoring=estimators, cv=5, n_jobs=-1, return_train_score=True)
print("\ncross validation scores:")
print("\nmean_absolute_error:{}".format(round(abs(cross_val["test_abs_error"]).mean()), 4), 
      "\nmean_squared_error:{}".format(round(abs(cross_val["test_squared_error"]).mean(), 4)),
      "\nroot_mean_squared_error:{}".format(round(math.sqrt(abs(cross_val["test_squared_error"]).mean()), 4)),
      "\nr2_score:{}".format(round(cross_val["test_r2_score"].mean(), 4)))

mae = round(mean_absolute_error(y_train, train_pred), 4)
mse = round(mean_squared_error(y_train, train_pred), 4)
rmse = round(math.sqrt(mse), 4)
r2 = round(r2_score(y_train, train_pred), 4)
print("\ntraining data scores:")
print("\nMAE_train:{}".format(mae),
      "\nMSE_train:{}".format(mse),
      "\nRMSE_train:{}".format(rmse),
      "\nr2_score_train:{}".format(r2))

#Saving the predicted data
y_pred_test = pd.DataFrame(test_pred, columns=list(target.columns))
y_pred_train = pd.DataFrame(train_pred, columns=list(target.columns))
train_pred = pd.DataFrame(scl_y.inverse_transform(y_pred_train), columns=list(target.columns))
test_pred = pd.DataFrame(scl_y.inverse_transform(y_pred_test), columns=list(target.columns))
fin_result = pd.concat([uniq_id, test_pred], axis=1)
fin_result.to_csv("finalsubmission.csv", encoding="utf-8", index=False)


