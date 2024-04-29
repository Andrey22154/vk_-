import numpy as np
import pandas as pd
import lightgbm as lgb

data = pd.read_parquet("path")

not_un = []

for col in data.drop(columns=['query_id', 'rank']).columns:
  if data[col].nunique() == 1:
    print(col, data[col].unique())
    not_un.append(col)

data = data.drop(columns=not_un)

corr_matrix = data.corr()

columns = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) >= 0.90:
            colname = corr_matrix.columns[i]
            columns.add(colname)

data = data.drop(columns=columns)

params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_at': [5],
    'learning_rate': 0.028588386237886294,
    'max_depth': 12,
    'num_leaves': 24,
    'min_data_in_leaf': 23,
    'num_iterations': 100,
    'random_state': 42,
    'lambdarank_norm': True,
    'verbose': -1
}
#сделаем имитацию прогноза, взяв последний запрос из датасета
last_query_id = data['query_id'].max()

train_data = data[data['query_id'] != last_query_id]
test_data = data[data['query_id'] == last_query_id]

lgb_train = lgb.Dataset(data=train_data.drop(['rank', 'query_id'], axis=1),
                        label=train_data['rank'],
                        group=train_data.groupby('query_id').size())

lgb_test = lgb.Dataset(data=test_data.drop(['rank', 'query_id'], axis=1),
                       label=test_data['rank'],
                       group=test_data.groupby('query_id').size())

model = lgb.train(params, lgb_train, valid_sets=[lgb_test])

test_preds = model.predict(test_data.drop(['rank', 'query_id'], axis=1))

predicted_ranks = test_preds.argsort()[::-1]
