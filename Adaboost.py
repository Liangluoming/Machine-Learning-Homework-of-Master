"""
@ author: Luoming Liang
@ email : liangluoming00@163.com
@ github: https://github.com/Liangluoming

"""


import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import warnings
warnings.filterwarnings('ignore')

## 读入数据
df = pd.read_csv("商业信贷数据tidy.csv")

## 读入变量数据(哪些变量用于分析)
x_list = pd.read_excel("X_list.xlsx")
X = df[x_list['features'].to_list()] ## 选择必要的列

## 一个贷款状态的映射字典
status_dic = {'Fully Paid' : 0, 'Charged Off' : 1, 'Late (31-120 days)' : 1, 'In Grace Period' : 1,
              'Late (16-30 days)' : 1, 'Default' : 1}
## 将字符串映射成数值
df['loan_status'] = df['loan_status'].map(status_dic)
y = df['loan_status']


## 划分训练集、验证集、测试集
x_train_valid, x_test, y_train_valid, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123,
                                                                stratify = y)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size = 0.1 ,random_state = 123,
                                                      stratify = y_train_valid)


def ada_gridsearch(param_dict, x_train, y_train, x_valid, y_valid, base_model, savefig = False, verbose = 9, random_state = 123):
    """
    Grid Search & Visualization
    Args:
        param_dict = {'learning_rate': list/array, 'n_estimators': list/array, 'fig_path' : str} : 用于存放学习率、学习器个数以及保存可视化图像路径的字典
        x_train, y_train : 训练集
        x_valid, y_valid : 验证集或测试集
        base_model : 基础学习器
        savefig : 保存可视化图的路径
        random_state : 随机种子
    Returns:
        accuracy : 不同参数下的精确率结果
        recall : 不同参数下的召回率结果
        f1scores : 不同参数下的F1score结果
    """
    accuracy = [[], [], [], []]  ## 用来存放不同参数下的accuracy
    recall = [[], [], [], []]  ## 用来存放不同参数下的recall
    f1scores = [[], [], [], []] ## 用来存放不同参数下的f1scores
    lr_dict = param_dict['learning_rate']
    n_estimators = param_dict['n_estimators']
    for idx, lr in enumerate(lr_dict):
        for n_estimator in n_estimators:
            if verbose == 1:
                print("lr:{}, n_estimator:{}".format(lr, n_estimator))
            ## 建立一个 Adaboost模型
            Ada_model = AdaBoostClassifier(base_estimator = base_model, n_estimators = n_estimator,learning_rate = lr, random_state = random_state)
            Ada_model.fit(x_train, y_train)  ## 在训练集上训练
            y_valid_pre = Ada_model.predict(x_valid) ## 用验证集去测试
            accuracy[idx] = np.append(accuracy[idx], Ada_model.score(x_valid, y_valid))  ## 保存accuracy
            recall[idx] = np.append(recall[idx], recall_score(y_valid, y_valid_pre))  ## 保存 precision
            f1scores[idx] = np.append(f1scores[idx], f1_score(y_valid, y_valid_pre))  ## 保存 f1score

    ## 可视化结果
    results = [accuracy, recall, f1scores] ## 用一个列表存放三个指标下的结果
    metrics_label = ['accuracy', 'call', 'f1 scores']  ## ylabel
    step = 1  ## 用来标记图的序号
    plt.figure(figsize = (12, 8)) ## 设置整个画布的大小
    ## 遍历结果 绘制 len(accuracy) * 3 个图
    for i in range(len(accuracy)):
        for j in range(3):
            plt.subplot(len(accuracy), 3, step)
            plt.plot(n_estimators, results[j][i])
            plt.ylabel(metrics_label[j])
            plt.grid()
            step += 1
    plt.tight_layout() ## 合理排布
    
    if savefig:
        path = param_dict['fig_path']
        plt.savefig(path, dpi = 1000)
    plt.show()
    return accuracy, recall, f1scores

## ======================================= ##
##             Grid Search 1               ##
## ======================================= ##
## 定义一个基础模型——数模型，树的最大深度为10
tree_model = DecisionTreeClassifier(max_depth = 10)
param_dict = {}
param_dict['learning_rate'] = [1e-3, 1e-2, 0.1, 1]  ## 学习率搜索范围
param_dict['n_estimators'] = np.arange(100, 1100, 100)  ## 分类器个数搜索范围
accuracy, recall, f1scores = ada_gridsearch(param_dict, x_train, y_train, x_valid, y_valid, tree_model, 123)  ## 网格搜索

## =============== brevity ================= ##

# tree_model = DecisionTreeClassifier(max_depth = 10)
# param_dict = {
#     'n_estimators' : np.arange(100, 1100, 100),
#     'learning_rate' : [1e-3, 1e-2, 0.1, 1, 2]
# }
# score_dict = {
#     'precision' : 'precision',         # 准确率
#     'recall' : 'recall',        # 召回率
#     'f1' : 'f1',
#     'roc_auc' : 'roc_auc'
# }
# Ada_model = AdaBoostClassifier(base_estimator = tree_model, random_state = 123)

# clf = GridSearchCV(Ada_model, 
#                    param_dict,
#                    cv = 10,
#                    scoring = score_dict,
#                    refit = 'f1',
#                    n_jobs = 8,
#                    verbose = 1)
# clf.fit(x_train_valid, y_train_valid)

n_estimators = param_dict['n_estimators']
print("Based accuracy:{}".format(n_estimators[np.argmax(accuracy[0])]))
print("Based recall:{}".format(n_estimators[np.argmax(recall[0])]))
print("Based f1_score:{}".format(n_estimators[np.argmax(f1scores[0])]))

## ======================================= ##
##             Grid Search 2               ##
## ======================================= ##

## 定义一个基础模型——数模型，树的最大深度为10
tree_model = DecisionTreeClassifier(max_depth = 10)
param_dict = {}
param_dict['learning_rate'] = [1e-4, 5e-4, 1e-3] ## 学习率搜索范围
param_dict['n_estimators'] = np.arange(20, 420, 20)  ## 分类器个数搜索范围
accuracy, recall, f1scores = ada_gridsearch(param_dict, x_train, y_train, x_valid, y_valid, tree_model, 123)  ## 网格搜索


## ======================================= ##
##             Grid Search 3               ##
## ======================================= ##

## 定义一个基础模型——数模型，树的最大深度为10
tree_model = DecisionTreeClassifier(max_depth = 10)
param_dict = {}
param_dict['learning_rate'] = [1e-6, 1e-5, 1e-4] ## 学习率搜索范围
param_dict['n_estimators'] = np.arange(10, 200, 10)  ## 分类器个数搜索范围
accuracy, recall, f1scores = ada_gridsearch(param_dict, x_train, y_train, x_valid, y_valid, tree_model, 123)  ## 网格搜索

n_estimators = param_dict['n_estimators']
print("Based accuracy:{}".format(n_estimators[np.argmax(accuracy[1])]))
print("Based recall:{}".format(n_estimators[np.argmax(recall[1])]))
print("Based f1_score:{}".format(n_estimators[np.argmax(f1scores[1])]))

## =========================================== ##
##    Identify best params & test dataset      ##
## =========================================== ##

## 根据网格搜索结果确定最优的分类器个数90，学习率0.00001
best_n_estimator = 90
best_learning_rate = 1e-5
tree_model = DecisionTreeClassifier(max_depth = 10)

## 构建Adaboost model
Ada_model = AdaBoostClassifier(base_estimator = tree_model, n_estimators = best_n_estimator,learning_rate = best_learning_rate, random_state = 123)
Ada_model.fit(x_train_valid, y_train_valid) ## 在训练集和验证集上训练
y_test_pre = Ada_model.predict(x_test) ## 在测试集上测试


print("accuracy of test set:{}".format(Ada_model.score(x_test, y_test)))
print("recall of test set:{}".format(recall_score(y_test, y_test_pre)))
print("f1 score of test set:{}".format(f1_score(y_test, y_test_pre)))

confusion_matrix(y_test, y_test_pre)  ## 混淆矩阵


## ========================== ##
##    feature importances     ##
## ========================== ##


Ada_model = AdaBoostClassifier(base_estimator = tree_model, n_estimators = best_n_estimator,learning_rate = best_learning_rate, random_state = 123)
Ada_model.fit(X, y) ## 在全样本上训练
y_pre = Ada_model.predict(X)

print("accuracy of test set:{}".format(Ada_model.score(X, y)))
print("recall of test set:{}".format(recall_score(y, y_pre)))
print("f1 score of test set:{}".format(f1_score(y, y_pre)))

confusion_matrix(y, y_pre)

## 查看各个变量的重要程度
features_coef = pd.DataFrame({'features' : Ada_model.feature_names_in_, 'coef' : Ada_model.feature_importances_}).sort_values(by = 'coef')

plt.figure(figsize = (12, 16))
plt.barh(y = features_coef['features'], width = features_coef['coef'])