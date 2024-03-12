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

label_encoder_geo = LabelEncoder()

# Encode the "Geography" column
df['geography_encoded'] = label_encoder_geo.fit_transform(df['Geography'])

# Drop the original "Geography" column
df.drop(columns=['Geography'], inplace=True)
label_encoder = LabelEncoder()
df['gender_encoded'] = label_encoder.fit_transform(df['Gender'])
df.drop(columns=['Gender'], inplace=True)
df


X = df.drop(columns=['RowNumber', 'Surname','Exited','CustomerId'], axis = 1)
# X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname','Geography','Gender','Exited','Age Group'], axis = 1)
# X.head(5)
Y = df["Exited"]

X = X.to_numpy()

Y = Y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)





# ## K nearest Classifier


<<<<<<< HEAD
knn=KNeighborsClassifier(n_neighbors=13)
=======
# Initialize the KNN model
knn=KNeighborsClassifier(n_neighbors=13)

# Train the model
knn.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_knn=knn.predict(X_test_scaled)

# Evaluate the model
accuracy_knn=accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy: {:.2f}%".format(accuracy_knn * 100))

# Classification report
print("\nKNN Classification Report:\n", classification_report(y_test, y_pred_knn))

# Confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("\nKNN Confusion Matrix:\n", cm_knn)
>>>>>>> 00a11680cb3024df3bb81b243cadef695b2cab55

# Train the model
knn.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_knn=knn.predict(X_test_scaled)

# Evaluate the model
accuracy_knn=accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy: {:.2f}%".format(accuracy_knn * 100))

# Classification report
print("\nKNN Classification Report:\n", classification_report(y_test, y_pred_knn))

# Confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("\nKNN Confusion Matrix:\n", cm_knn)











# ## Logistic Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#split the data into testing and training sets
<<<<<<< HEAD
# X_train, X_test, y_train, y_test = train_test_split(
#     X, Y, test_size=0.2, random_state=42)

#inbuilt function to train logistic regression model
Log_reg=LogisticRegression()
Log_reg.fit(X_train_scaled, y_train)

#accuracy 
y_pred = Log_reg.predict(X_test_scaled)
print("confusion_matrix\n", confusion_matrix(y_test, y_pred),"\n\n")
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
#classification report
print("\nClassification Report:\n", classification_report(y_test,y_pred))
=======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#inbuilt function to train logistic regression model
model=LogisticRegression()
model.fit(X_train, y_train)

#accuracy 
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
#classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
>>>>>>> 00a11680cb3024df3bb81b243cadef695b2cab55


# ## Support Vector mechine classifier (SVC)






# ## Decession Tree Classifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score

# Create a Decision Tree Classifier
dtc = DecisionTreeClassifier()

#Train the model
dtc.fit(X_train_scaled, y_train)

# Use the trained model to make predictions on the test set
y_pred_dt = dtc.predict(X_test_scaled)

#Calculate Accuracy Score
accuracy_dtc=accuracy_score(y_test,y_pred_dt)
print("Decision Tree Classifier Accuracy: {:.2f}%".format(accuracy_dtc * 100))

#classification report
print("\nClassification Report:\n", classification_report(y_test,y_pred_dt))

# Confusion matrix
cm_dtc = confusion_matrix(y_test, y_pred_dt)
print("\nDecision Tree Classifier Confusion Matrix:\n", cm_dtc)




# ## Random Forest Classifier





import pickle

# Suppose you have multiple models
models = {
    'Knn': knn,
    'LR': Log_reg,
    # 'svc': svc,
    # 'DTC':dtc,
    # 'rfc':rfc
}

# Dump the dictionary containing all models
with open('model.pkl', 'wb') as file:
    pickle.dump(models, file)

# Load the models from the file
with open('model.pkl', 'rb') as file:
    loaded_models = pickle.load(file)

# Access each model by its key
logistic_regression_model = loaded_models['LR']
# random_forest_model = loaded_models['rfc']
# svm_model = loaded_models['svc']
K_Nearest = loaded_models['Knn']
# Decision_tree = loaded_models['DTC']








