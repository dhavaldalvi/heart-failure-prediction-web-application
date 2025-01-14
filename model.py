import pandas as pd
from scipy import stats
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Loading the data
data = pd.read_csv("data/HeartFailurePredictionDataset.csv")

# Function to remove outliers
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

# Removing outliers from selected features
datadraft01 = remove_outlier(data, 'Cholesterol') 
df = remove_outlier(datadraft01, 'RestingBP') 

# Splitting dataset into training and testing dataset
X, y = df.drop(['HeartDisease'], axis = 1), df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classifying ordinal and nominal categorical and numerical data 
ordinal_cols = ['ST_Slope']
nominal_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina']
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

# Separately scaling and transforming the features 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Apply StandardScaler to numerical columns
        ('ord', OrdinalEncoder(), ordinal_cols),    # Apply OrdinalEncoder to ordinal columns
        ('nom', OneHotEncoder(), nominal_cols)      # Apply OneHotEncoder to nominal columns
    ])

# Making pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),           # Apply the preprocessing
    ('classifier', RandomForestClassifier())      # Train a classifier (e.g., Logistic Regression)
])

# Training the model
pipeline.fit(X_train, y_train)

# Saving the model
pickle.dump(pipeline, open('model.pkl', 'wb'))