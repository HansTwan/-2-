# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn import preprocessing
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular import FeatureMetadata
import warnings
warnings.filterwarnings('ignore')


######################### 模型一：无特征工程  ########################
train = pd.read_csv("../xfdata/电信客户流失预测挑战赛数据集/train.csv")
test = pd.read_csv("../xfdata/电信客户流失预测挑战赛数据集/test.csv")

data = pd.concat([train,test],axis=0,ignore_index = True)

features = [f for f in data.columns if f not in ['客户ID']]
train = data[data['是否流失'].notnull()].reset_index(drop=True)
test = data[data['是否流失'].isnull()].reset_index(drop=True)
train = train[features]
test = test[features]

train_data = TabularDataset(train)
predictor = TabularPredictor(label='是否流失', 
                             eval_metric='roc_auc', 
                             path='../user_data/model_data/model_1').fit(train_data,  
                                                                   num_bag_folds=5, 
                                                                   num_bag_sets=2, 
                                                                   num_stack_levels=2,   
                                                                   ag_args_fit={'num_gpus': 0},
                                                                   auto_stack=True,
                                                                   verbosity=2,
                                                                   presets='best_quality')
predictor.save_space() 

test_data = TabularDataset(test).drop('是否流失', axis=1)
prediction_model_1 = predictor.predict_proba(test_data)
#prediction_model_1

submission_model_1 = pd.read_csv('../user_data/submit_sample_data/sample_submit.csv')
submission_model_1['是否流失'] = prediction_model_1[1]
submission_model_1.to_csv('../user_data/submit_temp_data/submission_model_1.csv', index=False)


######################### 模型二：分箱  ########################
train = pd.read_csv("../xfdata/电信客户流失预测挑战赛数据集/train.csv")
test = pd.read_csv("../xfdata/电信客户流失预测挑战赛数据集/test.csv")
id ,label = '客户ID','是否流失'

for data in [train,test]:
    data['预计收入_bin'] = pd.qcut(data['预计收入'], 4, labels=False)
    data['平均月费用_bin'] = pd.qcut(data['平均月费用'], 10, labels=False)
    data['当前手机价格_bin'] = pd.cut(data['当前手机价格'], 10, labels=False)
    data['当前设备使用天数_bin'] = pd.qcut(data['当前设备使用天数'], 10, labels=False)
    data['客户生命周期内的总费用_bin'] = pd.qcut(data['客户生命周期内的总费用'], 10, labels=False)
    data['当月使用分钟数与前三个月平均值的百分比变化_bin'] = pd.qcut(data['当月使用分钟数与前三个月平均值的百分比变化'], 10, labels=False)
    data['客户生命周期内的平均每月使用分钟数_bin'] = pd.qcut(data['客户生命周期内的平均每月使用分钟数'], 10, labels=False)
    data['客户整个生命周期内的平均每月通话次数_bin'] = pd.qcut(data['客户整个生命周期内的平均每月通话次数'], 10, labels=False)
    data['客户生命周期内的总通话次数_bin'] = pd.qcut(data['客户生命周期内的总通话次数'], 100, labels=False)
    data['客户生命周期内的总使用分钟数_bin'] = pd.qcut(data['客户生命周期内的总使用分钟数'], 100, labels=False)
    data['客户生命周期内的总使用分钟数_bin'] = pd.qcut(data['客户生命周期内的总使用分钟数'], 100, labels=False)
    data['计费调整后的总费用_bin'] = pd.qcut(data['计费调整后的总费用'], 100, labels=False)
    data['计费调整后的总分钟数_bin'] = pd.qcut(data['计费调整后的总分钟数'], 100, labels=False)
    data['计费调整后的呼叫总数_bin'] = pd.qcut(data['计费调整后的呼叫总数'], 100, labels=False)
    
    
train_data = TabularDataset(train)

excluded_model_types = []
presets = ['best_quality']
predictor = TabularPredictor(label=label,
                             eval_metric = 'roc_auc',
                             path='../user_data/model_data/model_2').fit(train_data.drop(columns=[id]),
                                                                      num_bag_folds = 5,
                                                                      num_bag_sets = 3,
                                                                      num_stack_levels = 3,
                                                                 #     ag_args_fit = {'num_cpus':1},
                                                                      auto_stack = True,
                                                                      excluded_model_types=excluded_model_types,
                                                                      presets = presets)
predictor.save_space() 

test_data = TabularDataset(test)
prediction_model_2= predictor.predict_proba(test_data.drop(columns=[id]))
#rediction_model_2

submission_model_2 = pd.read_csv('../user_data/submit_sample_data/sample_submit.csv')
submission_model_2['是否流失'] = prediction_model_2[1]
submission_model_2.to_csv('../user_data/submit_temp_data/submission_model_2.csv', index=False)


################ 模型三：分箱+分组特征+业务特征+robust标准化 ###################
train = pd.read_csv("../xfdata/电信客户流失预测挑战赛数据集/train.csv")
test = pd.read_csv("../xfdata/电信客户流失预测挑战赛数据集/test.csv")
id ,label = '客户ID','是否流失'

for data in [train,test]:
    data['预计收入_bin'] = pd.qcut(data['预计收入'], 4, labels=False)
    data['平均月费用_bin'] = pd.qcut(data['平均月费用'], 10, labels=False)
    data['当前手机价格_bin'] = pd.cut(data['当前手机价格'], 10, labels=False)
    data['当前设备使用天数_bin'] = pd.qcut(data['当前设备使用天数'], 10, labels=False)
    data['客户生命周期内的总费用_bin'] = pd.qcut(data['客户生命周期内的总费用'], 10, labels=False)
    data['当月使用分钟数与前三个月平均值的百分比变化_bin'] = pd.qcut(data['当月使用分钟数与前三个月平均值的百分比变化'], 10, labels=False)
    data['客户生命周期内的平均每月使用分钟数_bin'] = pd.qcut(data['客户生命周期内的平均每月使用分钟数'], 10, labels=False)
    data['客户整个生命周期内的平均每月通话次数_bin'] = pd.qcut(data['客户整个生命周期内的平均每月通话次数'], 10, labels=False)
    data['客户生命周期内的总通话次数_bin'] = pd.qcut(data['客户生命周期内的总通话次数'], 100, labels=False)
    data['客户生命周期内的总使用分钟数_bin'] = pd.qcut(data['客户生命周期内的总使用分钟数'], 100, labels=False)
    data['客户生命周期内的总使用分钟数_bin'] = pd.qcut(data['客户生命周期内的总使用分钟数'], 100, labels=False)
    data['计费调整后的总费用_bin'] = pd.qcut(data['计费调整后的总费用'], 100, labels=False)
    data['计费调整后的总分钟数_bin'] = pd.qcut(data['计费调整后的总分钟数'], 100, labels=False)
    data['计费调整后的呼叫总数_bin'] = pd.qcut(data['计费调整后的呼叫总数'], 100, labels=False)
    
def get_features(df):
    features = [['客户生命周期内的总费用','计费调整后的总费用'],['客户生命周期内的总使用分钟数','计费调整后的总分钟数']]
    for fea in features:
        df[f'{fea[0]}_{fea[1]}_sub'] = df[fea[0]] - df[fea[1]]
    return df


for data in [train,test]:
    data = get_features(data)
    data['feature1'] = data['客户生命周期内的总使用分钟数'] / (data['客户生命周期内的总通话次数'] + 0.1)
    data['feature2'] = data['客户生命周期内的总费用'] / (data['客户生命周期内的总通话次数'] + 0.1)
    data['feature3'] = data['客户生命周期内的总费用'] / (data['客户生命周期内的总使用分钟数'] + 0.1)
    data['feature5'] = data['客户生命周期内的总费用'] / (data['当前设备使用天数'] + 0.1)
    data['feature6'] = data['客户生命周期内的总费用'] / (data['计费调整后的呼叫总数'] + 0.1)
    data['feature7'] = data['过去六个月的平均每月使用分钟数'] / (data['过去三个月的平均每月使用分钟数'] + 0.1)
    
    data['feature8'] = data['计费调整后的总费用'] / (data['客户生命周期内的总通话次数'] + 0.1)
    data['feature9'] = data['计费调整后的总费用'] / (data['客户生命周期内的总使用分钟数'] + 0.1)
    data['feature10'] = data['计费调整后的总费用'] / (data['当前设备使用天数'] + 0.1)
    data['feature11'] = data['计费调整后的总费用'] / (data['计费调整后的呼叫总数'] + 0.1)
    
for data in [train,test]:
    data['_调整后费用比']=data['计费调整后的总费用']/(data['客户生命周期内的总费用'] + 0.1)
    data['_超额使用分钟比']=data['平均超额使用分钟数']/(data['每月平均使用分钟数']+ 0.1)
    data['_客户生命周期月数'] = data['客户生命周期内的总费用']/(data['客户生命周期内平均月费用']+ 0.1)
    data['m_all'] = data['过去三个月的平均每月使用分钟数']/(data['客户整个生命周期内的平均每月通话次数']+ 0.1)

    data['mv_over'] = data['计费调整后的总分钟数'] / (data['计费调整后的呼叫总数']+ 0.1)
    data['6m_3m_使用分钟数']=data['过去六个月的平均每月使用分钟数']-data['过去三个月的平均每月使用分钟数']
    data['mv_sum'] = data['客户生命周期内的总通话次数'] / (data['客户生命周期内的总费用']+ 0.1)
    data['6m_3m_总通话次数']=data['过去六个月的平均每月通话次数']-data['过去三个月的平均每月通话次数']
    data['3m_天数'] = data['当前设备使用天数'] / (data['过去三个月的平均月费用']+ 0.1)
    data['mv_6m'] = data['过去六个月的平均每月使用分钟数'] / (data['过去六个月的平均月费用']+ 0.1)
    data['mvc_sum'] = data['客户生命周期内的总通话次数'] * data['客户生命周期内的总费用']
    data['all_天数'] = data['当前设备使用天数'] / (data['客户生命周期内的总费用']+ 0.1)
    data['mv_3m'] = data['过去三个月的平均每月使用分钟数'] / (data['过去三个月的平均月费用']+ 0.1)
    data['6m_天数'] = data['当前设备使用天数']*30 / (data['过去六个月的平均月费用']+ 0.1)
    data['over_天数'] = data['当前设备使用天数'] / (data['计费调整后的总费用']+ 0.1)
    data['6m_3m_平均月费用']=data['过去六个月的平均月费用']-data['过去三个月的平均月费用']

    data['gfo'] =data['使用高峰语音通话的平均不完整分钟数']*data['平均呼入和呼出高峰语音呼叫数']*0.5
    
    data['FM_3m'] = data['过去三个月的平均每月使用分钟数'] * data['过去三个月的平均每月通话次数'] / (data['过去三个月的平均月费用']+ 0.1)
    data['FM_6m'] = data['过去六个月的平均每月使用分钟数'] * data['过去六个月的平均每月通话次数'] / (data['过去六个月的平均月费用']+ 0.1)
    data['FM_sum'] = data['客户生命周期内的总通话次数'] * data['客户生命周期内的总使用分钟数'] / (data['客户生命周期内的总费用']+ 0.1)
    data['FM_over'] = data['计费调整后的总分钟数'] * data['计费调整后的呼叫总数'] / (data['计费调整后的总费用']+ 0.1)

    data['mvc_3m'] = data['过去三个月的平均每月使用分钟数'] * data['过去三个月的平均月费用']
    data['mvc_6m'] = data['过去六个月的平均每月使用分钟数'] * data['过去六个月的平均月费用']

    data['mvc_over'] = data['计费调整后的总分钟数'] * data['计费调整后的呼叫总数']
    data['语音完成比']=data['平均完成的语音呼叫数']*data['平均掉线或占线呼叫数']
    data['费用对经济情况的负担']=data['当前手机价格']*data['平均月费用']

    data['all_6m使用分钟数']=data['客户生命周期内的总使用分钟数']-data['过去六个月的平均每月使用分钟数']
    data['all_6m总通话次数']=data['客户生命周期内的总通话次数']-data['过去六个月的平均每月通话次数']
    data['all_6m平均月费用']=data['客户生命周期内的总费用']-data['过去六个月的平均月费用']

train2 = train
test2 = test

# robust标准化
robust = preprocessing.RobustScaler()
train = robust.fit_transform(train)
test = robust.fit_transform(test)

train = pd.DataFrame(train,index = train2.index,columns = train2.columns)
test = pd.DataFrame(test,index = test2.index,columns = test2.columns)

def adjust(x):
    if x >= 0:
        return 1
    else:
        return 0    
    
train['是否流失'] = train['是否流失'].apply(adjust)

train_data = TabularDataset(train)

excluded_model_types = []
presets = ['best_quality']
predictor = TabularPredictor(label=label,
                             eval_metric = 'roc_auc',
                             path='../user_data/model_data/model_3').fit(train_data.drop(columns=[id]),
                                                                      num_bag_folds = 5,
                                                                      num_bag_sets = 3,
                                                                      num_stack_levels = 3,
                                                                 #     ag_args_fit = {'num_cpus':1},
                                                                      auto_stack = True,
                                                                      excluded_model_types=excluded_model_types,
                                                                      presets = presets)
predictor.save_space() 

test_data = test #TabularDataset('test.csv')
prediction_model_3 = predictor.predict_proba(test_data.drop(columns=[id]))
#prediction_model_3

submission_model_3 = pd.read_csv('../user_data/submit_sample_data/sample_submit.csv')
submission_model_3['是否流失'] = prediction_model_3[1]
submission_model_3.to_csv('../user_data/submit_temp_data/submission_model_3.csv', index=False)


############################  模型融合  #####################################
#submission_model_1 0.97         模型一其次
#submission_model_2 0.97003+     模型二最好
#submission_model_3 0.95+        模型三最差


def ensemble_model_rank(result_1,result_2,weight_1,weight_2):
    result_1_pre = result_1['是否流失']
    result_2_pre = result_2['是否流失']
    last_result = pd.DataFrame({'客户ID': test2['客户ID'],
                               'result_1_pre': result_1_pre,
                               'result_2_pre': result_2_pre})
    last_result['result_1_pre'] = last_result['result_1_pre'].rank(pct=True)
    last_result['result_2_pre'] = last_result['result_2_pre'].rank(pct=True)
    last_result['是否流失'] = weight_1 * last_result['result_1_pre'] + weight_2 * last_result['result_2_pre']
    return last_result

def ensemble_model_weight(result_1,result_2,weight_1,weight_2):
    result_1_pre = result_1['是否流失']
    result_2_pre = result_2['是否流失']
    last_result = pd.DataFrame({'客户ID': test2['客户ID'],
                               'result_1_pre': result_1_pre,
                               'result_2_pre': result_2_pre})
    last_result['result_1_pre'] = last_result['result_1_pre']
    last_result['result_2_pre'] = last_result['result_2_pre']
    last_result['是否流失'] = weight_1 * last_result['result_1_pre'] + weight_2 * last_result['result_2_pre']
    return last_result

def get_prediction(last_result):
    prediction = last_result[['客户ID', '是否流失']]
    prediction.columns = ['客户ID', '是否流失']
    prediction.to_csv('../prediction_result/result.csv', index=False)
    return prediction

# 一层rank融合
rank_ensemble_1= ensemble_model_rank(submission_model_2,submission_model_3,0.7,0.3)  # 模型二与模型三七三开
rank_ensemble = ensemble_model_rank(rank_ensemble_1,submission_model_1,0.6,0.4)  # 基于上述结果与模型一六四开

# 一层weight融合,权重不变
weight_ensemble_1= ensemble_model_weight(submission_model_2,submission_model_3,0.7,0.3)  # 模型二与模型三七三开
weight_ensemble = ensemble_model_weight(weight_ensemble_1,submission_model_1,0.6,0.4)  # 基于上述结果与模型一六四开

# 二层rank融合
result = ensemble_model_rank(rank_ensemble,weight_ensemble,0.6,0.4)  # rank与weight六四开

# 输出结果
final_prediction = get_prediction(result)




