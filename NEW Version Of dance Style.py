# Import necessary libraries: 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Running app with finalized features:
# Selecting features (X) and target variable (y)
X_train = df1[['Height','Dance experience', 'Body type', 
         'Finding rhythm', 'Musical Instruments', 'Closet outfit', 
        'Partner','Dance steps Confusion', 'Flexibility', 'Describe yourself', 
        'Personal Distance','Cultures', 'Favorite genres', 'Dance Target']]
y_train = df1['Dance Type']

# Initialize LabelEncoder
label_encoders = {}

# Label encode categorical features
for col in X_train.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    X_train.loc[:, col] = label_encoders[col].fit_transform(X_train[col])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize the models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "GBoost": GradientBoostingClassifier(random_state=42),
    "NeuralNetwork": MLPClassifier(random_state=42, max_iter=300)
}

# Dictionary to hold the accuracy of each model
accuracy_scores = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_scores[name] = accuracy_score(y_test, y_pred)

# Convert the accuracy scores to a DataFrame
accuracy_df = pd.DataFrame(list(accuracy_scores.items()), columns=['Model', 'Accuracy'])
accuracy_df.sort_values(by='Accuracy', ascending=False, inplace=True)


# Test and Prediction
# Assuming you have the information about the person in a dictionary format

person_info = {
    'Height': '161-175cm',
    'Dance experience': '1 to 3 years of experience',
    'Body type': 'Naturally curvier or softer physique',
    'Finding rhythm': 'Most of the times I memorize the rhythm by practicing',
    'Musical Instruments': 'String instruments',
    'Closet outfit': 'Casual dress (T-shirt and pants)',
    'Partner': 'Yes',
    'Dance steps Confusion': 'Sometimes, but after 2-3 times it will be fine',
    'Flexibility': 'I almost have my splits and have a nice flexible back',
    'Describe yourself': 'Adventurous',
    'Personal Distance': 'Moderately Comfortable',
    'Cultures': 'Yes so much',
    'Favorite genres': 'Latino music',
    'Dance Target': 'Intermediate goal: Improving skills and dancing for self joy and self discovery', 
    
}

# Convert person_info into DataFrame
person_df = pd.DataFrame([person_info])

# Encode categorical features using label encoders
for col in person_df.select_dtypes(include=['object']).columns:
    person_df.loc[:, col] = label_encoders[col].transform(person_df[col])

# Make predictions with RandomForest Model
predicted_dance_type = rf_model.predict(person_df)

# Print the predicted dance type
print("Predicted Dance Type for the Person:")
print(predicted_dance_type[0])

