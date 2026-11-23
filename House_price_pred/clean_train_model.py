import pandas as pd
import numpy as np

data = pd.read_csv(r'D:\Learn_AI_&_ML\Projects\House_price_pred\dataset\Bengaluru_House_Data.csv')
# print(data.head())
# print(data.shape)
# print(data.info())
# for col in data.columns:
#     print(data[col].value_counts())
#     print("**" * 20)

# print(data.describe())
# 
# print(data.info())
# print(data['size'].value_counts())

data.drop(columns=['area_type', 'society', 'balcony', 'availability'], inplace=True)

data['size'] = data['size'].fillna('2 BHK')
data['bath'] = data['bath'].fillna(data['bath'].median())
# print(data.info())
data['location'] = data['location'].fillna('Other')
# print(data.isna().sum())
data['bhk'] = data['size'].str.split().str.get(0).astype(int)
# print(data[data.bhk > 20])
print(data['total_sqft'].unique())

def conver_range_to_num(x):
    temp = x.split('-')
    if len(temp) == 2:
        return (float(temp[0]) + float(temp[1])) / 2
    try:
        return float(x)
    except:
        return None
data['total_sqft'] = data['total_sqft'].apply(conver_range_to_num)
# print(data.head())
data['price_pre_sqft'] = data['price'] * 100000 / data['total_sqft']
# print(data['price_pre_sqft'])

data['location'] = data['location'].apply(lambda x: x.strip())
location_count = data['location'].value_counts()
location_count_less_10 = location_count[location_count <= 10]
# print(location_count_less_10)
data['location'] = data['location'].apply(lambda x: 'Other' if x in location_count_less_10 else x)
# print(data.describe())
# print(data['location'].value_counts())

# outlire removal
# print((data['total_sqft'] / data['bhk']).describe())

data = data[((data['total_sqft'] / data['bhk']) >= 300)]

# print((data.price_pre_sqft.describe()))

def remove_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_pre_sqft)
        st = np.std(subdf.price_pre_sqft)
        reduced_df = subdf[(subdf.price_pre_sqft > (m - st)) & (subdf.price_pre_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out
remove_outliers(data)
# print(data.shape)

def bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_pre_sqft),
                'std': np.std(bhk_df.price_pre_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_pre_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

bhk_outliers(data)
# print(bhk_outliers_removed.shape)


data.drop(columns=['size', 'price_pre_sqft'], inplace=True)
# print(data.head()   )

data.to_csv(r'D:\Learn_AI_&_ML\Projects\House_price_pred\dataset\cleaned_data.csv', index=False)

X = data.drop(columns=['price'])
y = data['price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

column_trans = make_column_transformer(
    (OneHotEncoder(sparse_output=False), ['location']),remainder='passthrough'
)

scaler = StandardScaler()
lr = LinearRegression()
pipeline = make_pipeline(column_trans, scaler, lr)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R2 score: {r2}")
# print("accuracy of model is: ", pipeline.score(X_test, y_test))
# print("Training R2:", pipeline.score(X_train, y_train))
# print("Testing R2:", pipeline.score(X_test, y_test)

# ===========================
# rf = RandomForestRegressor(
#     n_estimators=200,
#     max_depth=15,
#     min_samples_split=5,
#     random_state=42,
#     n_jobs=-1
# )
# rf_pipeline = make_pipeline(column_trans, scaler, rf)
# rf_pipeline.fit(X_train, y_train)
# y_pred_rf = rf_pipeline.predict(X_test)
# r2_rf = r2_score(y_test, y_pred_rf)
# print(f"R2 score (Random Forest): {r2_rf}")

# =============================

# from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
# ada_model = AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
grad_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
# xgb_model=XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)

# pipeline_xgb = make_pipeline(column_trans, scaler, xgb_model)
# pipeline_ada = make_pipeline(column_trans, scaler, ada_model)
pipeline_grad = make_pipeline(column_trans, scaler, grad_model)

# pipeline_ada.fit(X_train, y_train)
# y_pred_ada = pipeline_ada.predict(X_test)   
# r2_ada = r2_score(y_test, y_pred_ada)
# print(f"R2 score (AdaBoost): {r2_ada}")

pipeline_grad.fit(X_train, y_train)
y_pred_grad = pipeline_grad.predict(X_test)
r2_grad = r2_score(y_test, y_pred_grad)
print(f"R2 score (Gradient Boosting): {r2_grad}")

# pipeline_xgb.fit(X_train, y_train)
# y_pred_xgb = pipeline_xgb.predict(X_test)
# r2_xgb = r2_score(y_test, y_pred_xgb)
# print(f"R2 score (XGBoost): {r2_xgb}")
# print("R2 Score:", r2_score(y_test, y_pred_xgb))


# # lasso_model = Lasso()
# # pipeline_lasso = make_pipeline(column_trans, scaler, lasso_model)
# # pipeline_lasso.fit(X_train, y_train)
# # y_pred_lasso = pipeline_lasso.predict(X_test)
# # r2_lasso = r2_score(y_test, y_pred_lasso)
# # print(f"R2 score (Lasso): {r2_lasso}")

# # ridge_model = Ridge()
# # pipeline_ridge = make_pipeline(column_trans, scaler, ridge_model) 
# # pipeline_ridge.fit(X_train, y_train)  
# # y_pred_ridge = pipeline_ridge.predict(X_test)
# # r2_ridge = r2_score(y_test, y_pred_ridge)   
# # print(f"R2 score (Ridge): {r2_ridge}")

import pickle
pickle.dump(pipeline_grad, open(r'D:\Learn_AI_&_ML\Projects\House_price_pred\model\house_price_model_grad.pkl', 'wb'))