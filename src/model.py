"""
model.py — ML Model Training, Evaluation & Comparison
"""
import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix,
                              roc_auc_score)
from imblearn.over_sampling import SMOTE
import shap

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Features to exclude from training
EXCLUDE_COLS = [
    "date", "route_id", "route_name", "scheduled_departure", "actual_departure",
    "day_of_week", "event_types", "temp_category", "visibility_category"
]

# Target columns
REGRESSION_TARGET = "delay_minutes"
CLASSIFICATION_TARGET = "is_delayed"


def prepare_data(task="regression"):
    """Load encoded dataset and split into X, y."""
    filepath = os.path.join(PROCESSED_DIR, "encoded_dataset.csv")
    df = pd.read_csv(filepath)
    
    # Drop non-feature columns
    drop_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    
    if task == "regression":
        target = REGRESSION_TARGET
        extra_drop = ["is_delayed", "delay_category"] if "delay_category" in df.columns else ["is_delayed"]
    else:
        target = CLASSIFICATION_TARGET
        extra_drop = ["delay_minutes"]
    
    # Also drop any delay_category dummies for classification
    delay_cat_cols = [c for c in df.columns if c.startswith("delay_category_")]
    
    y = df[target]
    X = df.drop(columns=drop_cols + extra_drop + delay_cat_cols + [target], errors="ignore")
    
    # Drop any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Handle any remaining NaN
    X = X.fillna(0)
    
    print(f"📊 Data prepared for {task}: X={X.shape}, y={y.shape}")
    return X, y


def train_regression_models(X, y, test_size=0.2):
    """Train and compare regression models."""
    print("\n" + "="*60)
    print("📈 REGRESSION MODEL TRAINING")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=15),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6),
    }
    
    results = []
    best_score = -float("inf")
    best_model = None
    best_name = ""
    
    for name, model in models.items():
        print(f"🔄 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        
        results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2_Score": round(r2, 4),
            "CV_R2_Mean": round(cv_scores.mean(), 4),
            "CV_R2_Std": round(cv_scores.std(), 4),
        })
        
        print(f"   MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f} | CV_R²={cv_scores.mean():.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name
    
    results_df = pd.DataFrame(results)
    print(f"\n🏆 Best Regression Model: {best_name} (R²={best_score:.4f})")
    
    # Save best model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_regression_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"💾 Saved → {model_path}")
    
    # Save feature importance for tree-based models
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)
        fi.to_csv(os.path.join(PROCESSED_DIR, "feature_importance_regression.csv"), index=False)
        
        print("🧠 Computing SHAP values for best regression model...")
        try:
            X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer(X_sample)
            joblib.dump({"explainer": explainer, "shap_values": shap_values, "X_sample": X_sample}, 
                        os.path.join(PROCESSED_DIR, "shap_regression.pkl"))
            print("   ✅ SHAP values saved.")
        except Exception as e:
            print(f"   ⚠️ Could not compute SHAP values: {e}")
    
    return results_df, best_model, best_name, (X_train, X_test, y_train, y_test)


def train_classification_models(X, y, test_size=0.2):
    """Train and compare classification models."""
    print("\n" + "="*60)
    print("🎯 CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class distribution: {dict(y_train.value_counts())}\n")
    
    print("⚖️ Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Balanced distribution: {dict(pd.Series(y_train_res).value_counts())}\n")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=15, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6),
    }
    
    results = []
    best_score = -float("inf")
    best_model = None
    best_name = ""
    
    for name, model in models.items():
        print(f"🔄 Training {name}...")
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5, scoring="f1_weighted", n_jobs=-1)
        
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "CV_F1_Mean": round(cv_scores.mean(), 4),
            "CV_F1_Std": round(cv_scores.std(), 4),
        })
        
        print(f"   Acc={acc:.4f} | F1={f1:.4f} | CV_F1={cv_scores.mean():.4f}")
        
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name
    
    results_df = pd.DataFrame(results)
    print(f"\n🏆 Best Classification Model: {best_name} (F1={best_score:.4f})")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_classification_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"💾 Saved → {model_path}")
    
    # Classification report
    y_pred_best = best_model.predict(X_test)
    print(f"\n📋 Classification Report ({best_name}):")
    print(classification_report(y_test, y_pred_best))
    
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)
        fi.to_csv(os.path.join(PROCESSED_DIR, "feature_importance_classification.csv"), index=False)
        
        print("🧠 Computing SHAP values for best classification model...")
        try:
            X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer(X_sample)
            joblib.dump({"explainer": explainer, "shap_values": shap_values, "X_sample": X_sample}, 
                        os.path.join(PROCESSED_DIR, "shap_classification.pkl"))
            print("   ✅ SHAP values saved.")
        except Exception as e:
            print(f"   ⚠️ Could not compute SHAP values: {e}")
    
    return results_df, best_model, best_name, (X_train, X_test, y_train, y_test)


def run_full_training_pipeline():
    """Run both regression and classification pipelines."""
    print("="*60)
    print("🚀 FULL MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Regression
    X_reg, y_reg = prepare_data("regression")
    reg_results, reg_model, reg_name, reg_splits = train_regression_models(X_reg, y_reg)
    
    # Classification
    X_cls, y_cls = prepare_data("classification")
    cls_results, cls_model, cls_name, cls_splits = train_classification_models(X_cls, y_cls)
    
    # Save comparison tables
    reg_results.to_csv(os.path.join(PROCESSED_DIR, "regression_results.csv"), index=False)
    cls_results.to_csv(os.path.join(PROCESSED_DIR, "classification_results.csv"), index=False)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"\n📈 Regression:     {reg_name}")
    print(f"🎯 Classification: {cls_name}")
    
    return {
        "regression": {"results": reg_results, "model": reg_model, "name": reg_name, "splits": reg_splits},
        "classification": {"results": cls_results, "model": cls_model, "name": cls_name, "splits": cls_splits},
    }


if __name__ == "__main__":
    run_full_training_pipeline()
