import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(color_codes=True)
pd.set_option('display.max_columns', None)
df = pd.read_csv("supply_chain_data.csv")
print(df.head())

#data_processing

#type de données par colonne
# print(df.dtypes)
#afficher les colonnes ayant le type ("int64" par défaut) + object

print(df.select_dtypes(include="object").nunique())

print(df.shape)

#supprimer la colonne SKU (pas d'importance notable)
df.drop(columns='SKU',inplace=True)
print(df.nunique())

# list of categorical variables to plot
cat_vars = ['Product type', 'Customer demographics', 'Shipping carriers', 'Supplier name', 'Location', 
            'Inspection results', 'Transportation modes', 'Routes']

# create figure with subplots
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
axs = axs.flatten()

# create barplot for each categorical variable
for i, var in enumerate(cat_vars):
    sns.barplot(x=var, y='Costs', data=df, ax=axs[i], estimator=np.mean)
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)

# adjust spacing between subplots
fig.tight_layout()

# show plot
plt.show()
print(df.columns)
# num_vars = ['Price', 'Availability', 'Number of products sold','Stock levels','Lead times', 'Order quantities', 'Shipping times',
#        'Shipping costs', 'Supplier name', 'Location', 'Lead time','Production volumes', 'Manufacturing lead time', 'Manufacturing costs','Defect rates', 'Routes']
# fig, axs= plt.subplots(nrows=3, ncols=5, figsize= (20,10))
# axs = axs.flatten()

# for i, var in enumerate(num_vars):
#     sns.histplot(x=var, data=df, ax=axs[i])
# #remove the 14th subplot
# fig.delaxes(axs[13])
# #remove the 15th subplot
# fig.delaxes(axs[14])

# fig.tight_layout()
# plt.show()
check_missing=df.isnull().sum()*100/df.shape[0]
print("missing_values",check_missing[check_missing>0].sort_values(ascending=False))

#Label Encoding for Object Datatypes

#loop over each column in the DataFrame where dtype is "object"
for col in df.select_dtypes(include=["object"]).columns:
    #print the clumn name and the unique values
    print(f"{col}:{df[col].unique()}")
    
from sklearn import preprocessing
#boocler sur chaque colonne du DataFrame ou le dtype est "object"
for col in df.select_dtypes(include=["object"]).columns:
    #Initialize label encoder object
    label_encoder=preprocessing.LabelEncoder()
    #Fit the encoder to the unique values in the column
    label_encoder.fit(df[col].unique())
    #Transform the column using the encoder
    df[col]=label_encoder.transform(df[col])
    #Print the column name and the unique encoded values
    print(f"{col}:{df[col].unique()}")

#Correlation heatmap

plt.figure(figsize=(20,16))
sns.heatmap(df.corr(),fmt='.2g', annot=True)

#Train test split 

X = df.drop('Costs',axis=1)
y= df['Costs']

#20% test size and train 80% train size

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston

#Create a DecisionTreeRegressor object
dtree = DecisionTreeRegressor()

#Define the hyperparameters to tune and their values

param_grid= {
    'max_depth': [2,4,6,8]
    'min_samples_split': [2,4,6,8]
    'min_samples_leaf': [1,2,3,4]
    'max_feartures': ['auto', 'sqrt', 'log2']
    'random_state': [0,7,42]
    
    
    
}