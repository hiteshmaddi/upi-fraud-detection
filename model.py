import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset (sample for speed)
df = pd.read_csv("onlinefraud.csv").sample(100000, random_state=42)

# Drop unnecessary columns
if 'nameOrig' in df.columns:
    df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Features & target
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Convert categorical
X = pd.get_dummies(X)

# Handle imbalance
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print(report)

# Save everything
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))
pickle.dump(accuracy, open("accuracy.pkl", "wb"))