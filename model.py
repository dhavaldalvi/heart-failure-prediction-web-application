import pandas as pd
from scipy import stats
import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore') 

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

# Initializing the encoders for categorical features
Sex = LabelBinarizer()
ChestPainType = LabelBinarizer()
RestingECG = LabelBinarizer()
ExerciseAngina = LabelBinarizer()
ST_Slope = LabelBinarizer()

# Encoding the categorical features
df['Sex'] = Sex.fit_transform(df['Sex'])
df['ExerciseAngina'] = ExerciseAngina.fit_transform(df['ExerciseAngina'])
df = df.join(pd.DataFrame(ChestPainType.fit_transform(df["ChestPainType"]), columns=ChestPainType.classes_, index=df.index))
df = df.join(pd.DataFrame(RestingECG.fit_transform(df["RestingECG"]), columns=RestingECG.classes_, index=df.index))
df = df.join(pd.DataFrame(ST_Slope.fit_transform(df["ST_Slope"]), columns=ST_Slope.classes_, index=df.index))

# Finalized dataset for training
final_df = df.drop(['ChestPainType', 'RestingECG', 'ST_Slope'], axis=1)

# Splitting the dataset into training and testing sets
X, y = final_df.drop(['HeartDisease'], axis = 1), final_df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the training set and fitting the testing set
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Training the model
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
forest_prediction = forest.predict(X_test)
forest_accuracy = (round(accuracy_score(forest_prediction, y_test), 4)*100)
#print(f'Accuracy of the model is {forest_accuracy}')

pickle.dump(forest, open('model.pkl', 'wb'))
pickle.dump(Sex, open('feature_sex.pkl', 'wb'))
pickle.dump(ExerciseAngina, open('feature_exerciseangina.pkl', 'wb'))
pickle.dump(ChestPainType, open('feature_chestpain.pkl', 'wb'))
pickle.dump(RestingECG, open('feature_restingecg.pkl', 'wb'))
pickle.dump(ST_Slope, open('feature_stslope.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))