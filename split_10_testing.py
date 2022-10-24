#PyCharm 2020.1.5 (Community Edition)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
import random

#-----Reading the input data-----
colFeature  = pd.read_csv('T22.txt',sep='\t')
important_features = ['FAM19A5','TDRKH','KLK12','CNTNAP2','DCTN1','TRAF2','LY75','TNFRSF9','DCN','LAG3','LRRN1','IL10','METAP1D','CPXM1','CD4','SIGLEC6','TNFRSF4','CXCL9','ANGPT1','CD40-L','CBL','IRAK1']
target = 'Tendency'
y=colFeature[target]
Xs = pd.get_dummies(colFeature[important_features],drop_first=True)

#---- defining classification performance -----
def evaluateBinaryClassification(predictions, actuals):
    contigency = pd.crosstab(actuals, predictions)
    TP = contigency[1][1]
    TN = contigency[0][0]
    FP = contigency[1][0]
    FN = contigency[0][1]
    n = contigency.sum().sum()
    Acuracy = (TP + TN) / n
    Specificity = TN / (TN + FP)
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    return Acuracy, Recall, Precision, Specificity

#---- randomly select random_state for splitting dataset with all samples -----
num = range(1, 100)
random.seed(12)
nums = random.sample(num, 10)

print("-------------------XGBoost--------------------------------------")
acc=0
sens=0
spe=0
balance=0
roc=0
for i in nums:
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=i)
    XGB = xgb.XGBClassifier(n_estimators=100,
                             max_depth=2,
                             learning_rate=0.075,#
                             min_child_weight=1.8,
                             scale_pos_weight=5,
                             objective='binary:logistic',
                             random_state=8, use_label_encoder=False, eval_metric='mlogloss')
    XGB.fit(X_train, y_train)
    y_preds_class = XGB.predict(X_test)
    probas = XGB.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    roc=roc+roc_auc
    sens=sens+evaluateBinaryClassification(y_preds_class, y_test)[1]
    acc = acc + evaluateBinaryClassification(y_preds_class, y_test)[0]
    spe = spe + evaluateBinaryClassification(y_preds_class, y_test)[3]

acc=acc/10
sens=sens/10
spe=spe/10
balance=(sens+spe)/2
roc=roc/10
print("Sensitivity:",sens)
print("Specificity:",spe)
print("Accuracy:",acc)
print("ROC_AUC:",roc)
print("Balanced Accuracy:",balance)

print("-------------------ANN--------------------------------------")
acc=0
sens=0
spe=0
balance=0
roc=0
for i in nums:
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=i)
    reluMLP = MLPClassifier(activation='relu',
                            alpha=1e-4,
                            hidden_layer_sizes=(220,3),
                            learning_rate='constant',
                            learning_rate_init=0.001,
                            max_iter=1000,
                            shuffle=True,
                            solver='lbfgs',
                            random_state=8)

    reluMLP.fit(X_train, y_train)
    y_preds_class = reluMLP.predict(X_test)
    probas = reluMLP.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    roc = roc + roc_auc
    sens = sens + evaluateBinaryClassification(y_preds_class, y_test)[1]
    acc = acc + evaluateBinaryClassification(y_preds_class, y_test)[0]
    spe = spe + evaluateBinaryClassification(y_preds_class, y_test)[3]

acc=acc/10
sens=sens/10
spe=spe/10
balance=(sens+spe)/2
roc=roc/10
print("Sensitivity:",sens)
print("Specificity:",spe)
print("Accuracy:",acc)
print("ROC_AUC:",roc)
print("Balanced Accuracy:",balance)

print("-------------------DT--------------------------------------")
acc=0
sens=0
spe=0
balance=0
roc=0
for i in nums:
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=i)
    DT = DecisionTreeClassifier(criterion='entropy', max_depth=4, class_weight='balanced', random_state=8)
    DT.fit(X_train, y_train)
    y_preds_class = DT.predict(X_test)
    probas = DT.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    roc = roc + roc_auc
    sens = sens + evaluateBinaryClassification(y_preds_class, y_test)[1]
    acc = acc + evaluateBinaryClassification(y_preds_class, y_test)[0]
    spe = spe + evaluateBinaryClassification(y_preds_class, y_test)[3]

acc=acc/10
sens=sens/10
spe=spe/10
balance=(sens+spe)/2
roc=roc/10
print("Sensitivity:",sens)
print("Specificity:",spe)
print("Accuracy:",acc)
print("ROC_AUC:",roc)
print("Balanced Accuracy:",balance)

print("-------------------SVM--------------------------------------")
acc=0
sens=0
spe=0
balance=0
roc=0
for i in nums:
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=i)
    svm_cla = svm.SVC(C=1, kernel='rbf', class_weight='balanced',gamma='auto',random_state=8)
    SVC_ppp = svm.SVC(C=1, kernel='rbf', class_weight='balanced',gamma='auto',random_state=8, probability=True)
    svm_cla.fit(X_train, y_train)
    y_preds_class = svm_cla.predict(X_test)
    SVC_ppp.fit(X_train, y_train)
    probas = SVC_ppp.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    roc = roc + roc_auc
    sens = sens + evaluateBinaryClassification(y_preds_class, y_test)[1]
    acc = acc + evaluateBinaryClassification(y_preds_class, y_test)[0]
    spe = spe + evaluateBinaryClassification(y_preds_class, y_test)[3]

acc=acc/10
sens=sens/10
spe=spe/10
balance=(sens+spe)/2
roc=roc/10
print("Sensitivity:",sens)
print("Specificity:",spe)
print("Accuracy:",acc)
print("ROC_AUC:",roc)
print("Balanced Accuracy:",balance)

print("-------------------AdaBoost--------------------------------------")
acc=0
sens=0
spe=0
balance=0
roc=0
for i in nums:
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=i)
    lgb=AdaBoostClassifier(n_estimators=500,learning_rate=1,algorithm='SAMME',random_state=8)
    lgb.fit(X_train, y_train)
    y_preds_class = lgb.predict(X_test)
    probas = lgb.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    roc = roc + roc_auc
    sens = sens + evaluateBinaryClassification(y_preds_class, y_test)[1]
    acc = acc + evaluateBinaryClassification(y_preds_class, y_test)[0]
    spe = spe + evaluateBinaryClassification(y_preds_class, y_test)[3]

acc=acc/10
sens=sens/10
spe=spe/10
balance=(sens+spe)/2
roc=roc/10
print("Sensitivity:",sens)
print("Specificity:",spe)
print("Accuracy:",acc)
print("ROC_AUC:",roc)
print("Balanced Accuracy:",balance)