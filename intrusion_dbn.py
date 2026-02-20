import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data_path = "C:/Users/Vamshi/Desktop/Intrusion_detection/data/"

splits = [
    (f"{data_path}train_80.csv", f"{data_path}test_20.csv"),
    (f"{data_path}train_70.csv", f"{data_path}test_30.csv")
]

def preprocess_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

    categorical_cols = ['protocol_type', 'service', 'flag']

    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(combined.astype(str))
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

results_summary = []
detailed_path = f"{data_path}dbn_detailed_results.csv"

if os.path.exists(detailed_path):
    os.remove(detailed_path)

for train_file, test_file in splits:

    X_train, X_test, y_train, y_test = preprocess_data(train_file, test_file)

    rbm = BernoulliRBM(n_components=32, learning_rate=0.01, n_iter=20)
    logistic = LogisticRegression(max_iter=1000)
    dbn = Pipeline([('rbm', rbm), ('logistic', logistic)])

    dbn.fit(X_train, y_train)
    y_pred = dbn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"DBN Accuracy for {train_file}: {accuracy}")

    results_summary.append({'split': train_file, 'accuracy': accuracy})

    # -------- Power BI Detailed Export --------
    detailed_df = pd.DataFrame({
        "actual_label": y_test.values,
        "predicted_label": y_pred,
        "model_type": "DBN",
        "dataset_split": os.path.basename(train_file)
    })

    detailed_df["detection_status"] = np.where(
        detailed_df["actual_label"] == detailed_df["predicted_label"],
        "Correct",
        "Incorrect"
    )

    if hasattr(dbn, "predict_proba"):
        detailed_df["confidence_score"] = dbn.predict_proba(X_test).max(axis=1)
    else:
        detailed_df["confidence_score"] = 0.0

    if os.path.exists(detailed_path):
        detailed_df.to_csv(detailed_path, mode='a', header=False, index=False)
    else:
        detailed_df.to_csv(detailed_path, index=False)

pd.DataFrame(results_summary).to_csv(f"{data_path}dbn_results_comparison.csv", index=False)
print("✅ DBN detailed results exported for Power BI.")
