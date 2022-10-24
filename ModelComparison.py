#PyCharm 2020.1.5 (Community Edition)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


#-----Reading the input data-----
data = pd.read_csv('allsamples_importantFeatures.txt',sep='\t')
col = data.columns.values.tolist()
colFeature = col[1:23]
X = np.array(data[colFeature])
y = data['Tendency']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


#----- 5-fold cross-validation for five models-----
CV5=5

#----- Building XGBoost model -----
clf1 = xgb.XGBClassifier(n_estimators=100,
                         max_depth=2,
                         learning_rate=0.075,
                         min_child_weight=1.8,
                         scale_pos_weight=5,
                         objective='binary:logistic',
                         random_state=8, use_label_encoder=False, eval_metric='mlogloss')

#----- Calculating xgb model performance -----
predicted_prob_xgb = cross_val_predict(clf1, X_train, y_train, cv=CV5, method='predict_proba')[:,1]
print("XGBoost AUC: {}".format(roc_auc_score(y_train, predicted_prob_xgb)))
yhat = predicted_prob_xgb
pos_pred1 = predicted_prob_xgb

#----- Building ANN model -----
clf1 = MLPClassifier(activation='relu',
                     alpha=1e-4,
                     hidden_layer_sizes=(220,3),
                     learning_rate='constant',
                     learning_rate_init=0.001,
                     max_iter=1000,
                     shuffle=True,
                     solver='lbfgs',
                     random_state=8)

#----- Calculating ANN model performance -----
predicted_prob = cross_val_predict(clf1, X_train, y_train, cv=CV5, method='predict_proba')[:,1]
print("ANN AUC: {}".format(roc_auc_score(y_train, predicted_prob)))
yhat = predicted_prob
pos_pred2 = predicted_prob

#----- Building DT model -----
clf1 = DecisionTreeClassifier(criterion='entropy', max_depth=4,class_weight='balanced',random_state=8)

#----- Calculating DT model performance -----
predicted_prob = cross_val_predict(clf1, X_train, y_train, cv=CV5, method='predict_proba')[:,1]
#print(predicted_prob)
print("DT AUC: {}".format(roc_auc_score(y_train, predicted_prob)))

yhat = predicted_prob
pos_pred3 = predicted_prob

#----- Building SVM model -----
clf1 = svm.SVC(C=1, kernel='rbf', class_weight='balanced',gamma='auto',random_state=8)

#----- Calculating SVM model performance -----

clf1=svm.SVC(C=1, kernel='rbf', class_weight='balanced',gamma='auto',random_state=8, probability=True)
predicted_prob = cross_val_predict(clf1, X_train, y_train, cv=CV5, method='predict_proba')[:,1]
print("SVM AUC: {}".format(roc_auc_score(y_train, predicted_prob)))

yhat = predicted_prob
pos_pred4 = predicted_prob

#----- Building AdaBoost model -----
clf1 = AdaBoostClassifier(n_estimators=500,learning_rate=1,algorithm='SAMME',random_state=8)

#----- Calculating AdaBoost model performance -----
predicted_prob = cross_val_predict(clf1, X_train, y_train, cv=CV5, method='predict_proba')[:,1]
print("AdaBoost AUC: {}".format(roc_auc_score(y_train, predicted_prob)))

yhat = predicted_prob
pos_pred5 = predicted_prob

#----- Drawing ROC curves -----
plt.plot([0,1], [0, 1], linestyle='--', label='random')
fpr1, tpr1, _ = roc_curve(y_train, pos_pred1)
fpr2, tpr2, _ = roc_curve(y_train, pos_pred2)
fpr3, tpr3, _ = roc_curve(y_train, pos_pred3)
fpr4, tpr4, _ = roc_curve(y_train, pos_pred4)
fpr5, tpr5, _ = roc_curve(y_train, pos_pred5)
plt.plot(fpr1, tpr1, marker='.', label='XGBoost')
plt.plot(fpr4, tpr4, marker='.', label='SVM')
plt.plot(fpr5, tpr5, marker='.', label='AdaBoost')
plt.plot(fpr2, tpr2, marker='.', label='ANN')
plt.plot(fpr3, tpr3, marker='.', label='DT')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
