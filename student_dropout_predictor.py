# student_dropout_predictor.py

import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import classification_report, accuracy_score # type: ignore

# Load dataset
df = pd.read_csv("xAPI-Edu-Data.csv")

# Encode categorical columns
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Define X and y
X = df.drop("Class", axis=1)
y = df["Class"]  # Dropout = 0, Enrolled = 1, Graduate = 2 (after encoding)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
