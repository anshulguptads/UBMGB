import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
)
from sklearn.pipeline import Pipeline
    # ColumnTransformer and OneHotEncoder allow flexibility if any column is non-numeric later.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

RANDOM_STATE = 42

def split_X_y(df, drop_zip=False):
    df = df.copy()
    id_cols = [c for c in df.columns if c.lower().replace("_","").replace(" ","") in ["id","customerid","custid"]]
    for c in id_cols:
        if c in df.columns: df.drop(columns=[c], inplace=True)
    target = "Personal Loan"
    X = df.drop(columns=[target])
    if drop_zip and "Zip code" in X.columns:
        X = X.drop(columns=["Zip code"])
    y = df[target]
    cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return X, y, cat_cols, num_cols

def make_preprocessor(cat_cols, num_cols):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

def get_models(which):
    models = {}
    if "Decision Tree" in which:
        models["Decision Tree"] = DecisionTreeClassifier(random_state=RANDOM_STATE)
    if "Random Forest" in which:
        models["Random Forest"] = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    if "Gradient Boosted Trees" in which:
        models["Gradient Boosted Trees"] = GradientBoostingClassifier(random_state=RANDOM_STATE)
    return models

def train_and_evaluate(df, which_models, drop_zip=False):
    X, y, cat_cols, num_cols = split_X_y(df, drop_zip=drop_zip)
    preprocess = make_preprocessor(cat_cols, num_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    results = []
    fitted = {}
    roc_data = {}

    for name, model in get_models(which_models).items():
        pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

        y_tr_pred = pipe.predict(X_train)
        y_te_pred = pipe.predict(X_test)

        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_te_score = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe.named_steps["model"], "decision_function"):
            y_te_score = pipe.decision_function(X_test)
        else:
            y_te_score = None

        metrics = {
            "Model": name,
            "Training Accuracy": accuracy_score(y_train, y_tr_pred),
            "Testing Accuracy": accuracy_score(y_test, y_te_pred),
            "Precision": precision_score(y_test, y_te_pred, zero_division=0),
            "Recall": recall_score(y_test, y_te_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_te_pred, zero_division=0),
        }
        results.append(metrics)

        cm = confusion_matrix(y_test, y_te_pred)
        fitted[name + " Confusion Matrix"] = cm

        if y_te_score is not None:
            fpr, tpr, _ = roc_curve(y_test, y_te_score)
            roc_auc = auc(fpr, tpr)
            roc_data[name] = (fpr, tpr, roc_auc)

    results_df = pd.DataFrame(results)
    return fitted, results_df, roc_data, (X_train, X_test, y_train, y_test)

def predict_new(pipe, row_dict, df_reference, drop_zip=False):
    input_df = pd.DataFrame([row_dict])
    pred = int(pipe.predict(input_df)[0])
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        proba = float(pipe.predict_proba(input_df)[0, 1])
    elif hasattr(pipe.named_steps["model"], "decision_function"):
        score = pipe.decision_function(input_df)[0]
        proba = float(1.0 / (1.0 + np.exp(-score)))
    else:
        proba = None
    return pred, proba
