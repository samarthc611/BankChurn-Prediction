#!/usr/bin/env python
# coding: utf-8

# If these libraries like ***Seaborn, Matplotlib, Pandas, NumPy, Scikit-learn, TensorFlow*** are not installed on your PC, please uncomment the next line to install them.

# In[166]:


# !pip install -q seaborn
# !pip install -q matplotlib
# !pip install -q pandas
# !pip install -q numpy
# !pip install -q sklearn
# # For GPU users
# !pip install -q tensorflow[and-cuda]
# # For CPU users
# !pip install -q tensorflow
# get_ipython().system('pip install pickle')


# In[167]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, classification_report, confusion_matrix

# get_ipython().run_line_magic('matplotlib', 'inline')


# #### Data reading using pandas 

# In[168]:


df = pd.read_csv('./churn.csv', delimiter=',')


# In[ ]:





# ## Data Visualisation 

# In[169]:


# sns.countplot(x='Exited', data=df)
# plt.title('Count of Churn')
# plt.show()


# In[170]:


# sns.countplot(data=df, x="NumOfProducts", hue="Exited")
# plt.show()


# In[171]:


# sns.displot(data=df, x="Balance", hue="Exited", kde=True)


# In[172]:


# plt.figure(figsize=(15,6))
# sns.displot(data=df, x="CreditScore", hue="Exited", kde=True)
# plt.show()


# In[173]:


# plt.figure(figsize=(15,6))
# sns.countplot(data=df, x="Tenure", hue="Exited")
# plt.show()


# In[174]:


# sns.violinplot(data=df, y="EstimatedSalary",x="Exited", hue="Exited")


# In[175]:


# value_count = df["Exited"].value_counts()

# plt.figure(figsize=(6, 6))
# plt.pie(value_count, labels=['Non-Churned', 'Churned'], autopct='%1.1f%%', colors=['Cyan', 'pink'])
# plt.title('Proportion of Churned vs. Non-Churned Customers')
# plt.show()



# In[176]:


# sns.countplot(x='Gender', hue='Exited', data=df)
# plt.title('Churn by Gender')
# plt.show()

# sns.countplot(x='Geography', hue='Exited', data=df)
# plt.title('Churn by Geography')
# plt.show()

# # 3. Bar Plot of Churn by Age Group
# df['Age Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 70, 100], labels=['0-30', '31-40', '41-50', '51-60', '61-70', '71-100'])
# sns.countplot(x='Age Group', hue='Exited', data=df)
# plt.title('Churn by Age Group')
# plt.show()


# # 4. Box Plot of Balance by Churn
# sns.boxplot(x='Exited', y='Balance', data=df)
# plt.title('Balance Distribution by Churn')
# plt.show()



# In[177]:


# # 5. Pair Plot of Numeric Variables by Churn
# numeric_data = df.select_dtypes(include=['int64', 'float64'])
# sns.pairplot(data=numeric_data, hue='Exited')
# plt.title('Pair Plot of Numeric Variables by Churn')


# In[178]:


# sns.scatterplot(x='Age', y='Balance', hue='Exited', data=df)
# plt.title('Scatter Plot of Balance vs. Age, colored by Churn')
# plt.show()


# In[179]:


# sns.boxplot(x='Gender', y='Balance', hue='Exited', data=df)
# plt.title('Box Plot of Balance vs. Gender, separated by Churn')
# plt.show()


# In[180]:


# sns.pairplot(data=df, vars=['Age', 'Balance', 'EstimatedSalary'], hue='Exited')
# plt.show()


# ## Feature Engineering 

# In[181]:


# len(df.columns)


# In[182]:


label_encoder_geo = LabelEncoder()

# Encode the "Geography" column
df['geography_encoded'] = label_encoder_geo.fit_transform(df['Geography'])

# Drop the original "Geography" column
df.drop(columns=['Geography'], inplace=True)
label_encoder = LabelEncoder()
df['gender_encoded'] = label_encoder.fit_transform(df['Gender'])
df.drop(columns=['Gender'], inplace=True)
df


# In[183]:


X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname','Exited'], axis = 1)
# X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname','Geography','Gender','Exited','Age Group'], axis = 1)
# X.head(5)
Y = df["Exited"]


# In[184]:


X = X.to_numpy()
X


# In[185]:


Y = Y.to_numpy()
Y


# In[186]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[187]:


X_train


# In[188]:


X_train[0]


# In[189]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# .

# In[ ]:





# ## K nearest Classifier

# In[190]:


knn = KNeighborsClassifier(n_neighbors=7)


# In[191]:


knn.fit(X_train_scaled, y_train)


# In[192]:


y_pred = knn.predict(X_test_scaled)


# In[193]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[194]:


error_rate = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_scaled, y_train)
    pred_i = knn.predict(X_test_scaled)
    error_rate.append(np.mean(y_test != pred_i))


# In[195]:


# plt.figure(figsize=(15,6))
# plt.plot(np.arange(1,20), error_rate, color='blue', linestyle='dashed', marker='o',
#          markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value')
# plt.xticks(np.arange(0, 20, 1.0))
# plt.xlabel('K')
# plt.ylabel('Error Rate')


# In[196]:


# We concludede that from error rate that using arm method that 2 is point where maximum error


# In[197]:


knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train_scaled, y_train)


# In[198]:


predict = knn.predict(X_test_scaled)

# print("confusion_matrix\n", confusion_matrix(y_test, predict),"\n\n")
# print("classification_report\n\n",classification_report(y_test, predict),"\n\n")
# print("accuracy_score = ", accuracy_score(y_test, predict), "\n")


# ## Logistic Regression

# In[199]:


Log_reg = LogisticRegression()
Log_reg.fit(X_train_scaled, y_train)


# In[200]:


predict = Log_reg.predict(X_test_scaled)

print("confusion_matrix\n", confusion_matrix(y_test, predict),"\n\n")
print("classification_report\n\n",classification_report(y_test, predict),"\n\n")
print("accuracy_score = ", accuracy_score(y_test, predict), "\n")


# ## Support Vector mechine classifier (SVC)

# In[201]:


svc = SVC()
svc.fit(X_train_scaled, y_train)


# In[202]:


predict = svc.predict(X_test_scaled)

print("confusion_matrix\n", confusion_matrix(y_test, predict),"\n\n")
print("classification_report\n\n",classification_report(y_test, predict),"\n\n")
print("accuracy_score = ", accuracy_score(y_test, predict), "\n")


# ## Decession Tree Classifier 

# In[203]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train_scaled, y_train)


# In[204]:


predict = dtc.predict(X_test_scaled)

print("confusion_matrix\n", confusion_matrix(y_test, predict),"\n\n")
print("classification_report\n\n",classification_report(y_test, predict),"\n\n")
print("accuracy_score = ", accuracy_score(y_test, predict), "\n")


# ## Random Forest Classifier

# In[205]:


rfc = RandomForestClassifier(n_estimators=250)
rfc.fit(X_train_scaled, y_train)


# In[206]:


predict = rfc.predict(X_test_scaled)

print("confusion_matrix\n", confusion_matrix(y_test, predict),"\n\n")
print("classification_report\n\n",classification_report(y_test, predict),"\n\n")
print("accuracy_score = ", accuracy_score(y_test, predict), "\n")


import pickle

# Suppose you have multiple models
models = {
    'Knn': knn,
    'LR': Log_reg,
    'svc': svc,
    'DTC':dtc,
    'rfc':rfc
}

# Dump the dictionary containing all models
with open('model.pkl', 'wb') as file:
    pickle.dump(models, file)

# Load the models from the file
with open('model.pkl', 'rb') as file:
    loaded_models = pickle.load(file)

# Access each model by its key
logistic_regression_model = loaded_models['LR']
random_forest_model = loaded_models['rfc']
svm_model = loaded_models['svc']
K_Nearest = loaded_models['knn']
Decision_tree = loaded_models['DTC']





# In[ ]:




