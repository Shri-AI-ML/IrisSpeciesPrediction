import pandas as pd 
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np 



iris = load_iris()\
    
X = pd.DataFrame(iris.data , columns=iris.feature_names)
X.head()

y= pd.Series(iris.target)   #name of flowers asign as number 
y

df = X.copy()
df['target']=y
df.head    

sns.countplot(x=df.target)
plt.title("Class Distribution in Iris Dataset")
plt.xlabel("Target_class")
plt.ylabel("Count")                  #count distribution plot 



sns.pairplot(df,hue = 'target')                               #Pairwise feature relationship 
plt.suptitle("Pairplot of Iris Feature", y = 1.02)
plt.show()

#Correlation Map 
sns.heatmap(df.corr(),annot= True,cmap = "coolwarm")
plt.title("Feature Corelation Heatmap")
plt.show()


 
#Data proccessing 
 
X_train,X_test, y_train , y_test = train_test_split(X,y,test_size=0.2,random_state = 42)              #split the data 


#Scaling the data set 

scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)     #mean = 0 & Standard deviation = 1
X_test= scaler.fit_transform(X_test)

#Data set before and after scaling 



fig,ax = plt.subplots(1,2,figsize=(14,5))

ax[0].hist(X.iloc[:,0],bins = 20 ,color= "skyblue")
ax[0].set_title("Before Scaling : Sepal Length")

ax[1].hist(X_train[:,0],bins = 20 ,color= "orange")
ax[0].set_title("After Scaling : Sepal Length")
plt.show()

#K-Nearest Neighbors(KNN)

from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)


#Decision-Tree

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4,random_state= 42)

tree.fit(X_train,y_train)
y_pred_tree = tree.predict(X_test)


#Visiualizing Model Evalution 

from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion(y_true,y_pred,title):
    cm = confusion_matrix(y_true,y_pred)
    sns.heatmap(cm,annot = True , fmt='d',cmap='Blues')
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title("Title")
    plt.show()
    
    

plot_confusion(y_test,y_pred_knn,"KNN Confusion Matrix")
plot_confusion(y_test,y_pred_tree,"Decision Tree Confusion Matrix ")    