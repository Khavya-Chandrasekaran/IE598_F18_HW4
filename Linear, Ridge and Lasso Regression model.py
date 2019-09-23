#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('K:\Fall 2019\MLF\Assignments\HW4\housing2.csv')
df_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()
row=len(df.index)
col=len(df.columns)
print("The number of rows are "+str(row))
print("The number of columns are "+str(col))

#Statistics Summary
stat=df.describe()
print("The Statistics Summary is as follows")
print(stat)

median=["MEDV"]
RM=['RM']

#Correlation between Classification Target and Real Attributes
from random import uniform
target=df[median]
attribute=df[RM]
plt.scatter(attribute, target)
plt.xlabel("Attribute- Average number of rooms")
plt.ylabel("Target- Median Value")
print("Plot of the Correlation between Classification Target and Real Attributes")
plt.show()

#Presenting Attribute Correlations Visually
from pandas import DataFrame
corMat = DataFrame(df.corr())

#visualize correlations using heatmap
plt.pcolor(corMat)
print("Heat Map1")
plt.show()


# In[42]:


import seaborn as sns
sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()


# In[43]:


import numpy as np
corr = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
heatmap = sns.heatmap(corr, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()


# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()
print(lr.score(X_test,y_test))
print('Slope: %.3f' % lr.coef_[0])
print('Intercept: %.3f' % lr.intercept_)

from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))

from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


# In[56]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []
ridge = Ridge(normalize=True)
for alpha in alpha_space:
    ridge.alpha = alpha   
    ridge_cv_scores = cross_val_score(ridge,X,y,cv=10)
    ridge_scores.append(np.mean(ridge_cv_scores))
    ridge_scores_std.append(np.std(ridge_cv_scores))
plt.plot(ridge_scores, ridge_scores_std)
plt.xlabel("Alpha")
plt.ylabel("CV Score")
plt.show()
ridge.fit(X_train,y_train)

ridge_coef = ridge.coef_
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()
print('Slope: %.3f' % ridge.coef_[0])
print('Intercept: %.3f' % ridge.intercept_)
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
print("Ridge Regression Model Coefficients= " + str(ridge_coef))
print("Ridge Regression Model Accuracy= " + str(ridge.score(X_test, y_test)))

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1) 
lasso_cv_scores = cross_val_score(lasso,X,y,cv=5)
print("Lasso Cross Validation Accuracy= "+str(np.mean(lasso_cv_scores)))
lasso.fit(X_train,y_train)
lasso_coef = lasso.coef_
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()
print('Slope: %.3f' % lasso.coef_[0])
print('Intercept: %.3f' % lasso.intercept_)
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
print("Lasso Regression Model Coefficients= " +str(lasso_coef))
print("Lasso Regression Model Accuracy= " + str(lasso.score(X_test, y_test)))

from sklearn.linear_model import ElasticNet
lasso = ElasticNet(alpha=0.1, l1_ratio=0.5)
lasso.fit(X_train,y_train)
lasso_coef = lasso.coef_
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()
print('Slope: %.3f' % lasso.coef_[0])
print('Intercept: %.3f' % lasso.intercept_)
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
print("ElasticNet Lasso Regression Model Coefficients= " +str(lasso_coef))
print("ElasticNet Lasso Regression Model Accuracy= " + str(lasso.score(X_test, y_test)))

print("My name is Khavya Chandrasekaran")
print("My NetID is: khavyac2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:




