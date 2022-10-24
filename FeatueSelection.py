#PyCharm 2020.1.5 (Community Edition)
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve
import shap

#-----Reading the input data-----
dataset = pd.read_csv('matched.txt',sep='\t')
col = dataset.columns.values.tolist()
varFeatures = col[1:278]
data_x = np.array(dataset[varFeatures])
data_y = dataset['Tendency']

#----- Building model ------
xgbModel = xgb.XGBClassifier(n_estimators=40,max_depth=2,learning_rate=0.01,colsample_bytree=0.7,objective='binary:logistic',random_state=8,use_label_encoder =False, eval_metric='mlogloss')

#----- Calculating model performance -----
cv = LeaveOneOut()
predicted = cross_val_predict(xgbModel, data_x, data_y,cv=cv)
predicted_prob = cross_val_predict(xgbModel, data_x, data_y, cv=cv, method='predict_proba')[:,1]
print("Overall accuracy : {}".format(metrics.accuracy_score(predicted, data_y)))
print("ROC-AUC score: {}".format(roc_auc_score(data_y, predicted_prob)))

#----- Drawing ROC curve -----
xgbModel.fit(data_x, data_y)
yhat = predicted_prob
pos_pred = predicted_prob
plt.plot([0,1], [0, 1], linestyle='--', label='random')
fpr, tpr, _ = roc_curve(data_y, pos_pred)
plt.plot(fpr, tpr, marker='.', label='XGBoost')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

#----- showing all important features -----
xgbModel.fit(data_x, data_y)
explainer = shap.TreeExplainer(xgbModel)
shap_values = explainer.shap_values(data_x)
allFeatures=pd.DataFrame(data_x, columns = dataset.columns.drop('Tendency'))
shap.summary_plot(shap_values, allFeatures, plot_type="bar", max_display=22)
