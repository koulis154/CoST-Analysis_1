
# In[1]:


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import warnings 
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import statsmodels.api as sm
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.svm import SVR
from sklearn import decomposition
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier


# In[2]:


# we read the dataset 
df_data = pd.read_csv("/Users/kyriakoskyriazis/Desktop/Rsearch Workshop/CoST/CoST.csv")
# we drop na
df_data = df_data.dropna()
# we reset the index after we drop na, so we wont encanter any problems 
df_data = df_data.reset_index(drop=True)
# we drop the variable "gesture" that's the variable we gonna use for our predictions  
X = df_data.drop(' variant', axis = 1)
# we grab the variable "gesture" so we can use it for our predictions 
y = df_data[' variant']
# we split the data set into "tarin", "test" and "validation" set. the ration between "train" and "test" is 30% and the ratio between "test" and "validation" set is 50%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X)
print(df_data.keys())



# In[3]:


#criterion = 'entropy', max_depth = 30, random_state = 0
tree = DecisionTreeClassifier()
# here we tarin our model 
tree.fit(X_train, y_train)
# here we try to predict the values of the "X_test" set from our data set 
tree_predict = tree.predict(X_test)
# here we fit the prediction model we created and the "y_test" set from our data set, and we get the accuracy score
score = accuracy_score(tree_predict, y_test)
 
print("Training set score: {:.3f}".format(tree.score(X_train, y_train))) 
print("Test set score: {:.3f}".format(tree.score(X_test, y_test)))
print("Cross Validation score", round(cross_val_score(tree, X, y, cv=5).mean(), 3))


# In[6]:




mtx = confusion_matrix(tree_predict, y_test)
cfmap= sns.heatmap(mtx, cmap = "Blues", annot = True, fmt = "d")

plt.title('')
plt.ylabel('')
plt.xlabel('')
plt.show()




# In[10]:



forest = RandomForestClassifier(n_jobs = -1)
# here we train our random forest model using the "training" set 
forest.fit(X_train, y_train)
# here we try to predict the values of the "X_test" set from our data set 
forest_pred = forest.predict(X_test)
# here we fit the prediction model we created and the "y_test" set from our data set, and we get the accuracy score
forest_score = accuracy_score(forest_pred, y_test)

print("Training set score: {:.3f}".format(tree.score(X_train, y_train))) 
print("Test set score: {:.3f}".format(tree.score(X_test, y_test)))
print("Cross Validation score", round(cross_val_score(forest, X, y, cv=5).mean(), 3))


# In[9]:



mtx_1 = confusion_matrix(forest_pred, y_test)
cfmap = sns.heatmap(mtx_1, cmap = "Blues", annot = True, fmt = "d")

plt.title('')
plt.ylabel('')
plt.xlabel('')
plt.show()

