from __future__ import annotations
import numpy as np
import pandas as pd
from functools import partial
import warnings
from typing import List, Tuple, Sequence, Union

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, confusion_matrix
)
from xgboost import XGBClassifier

import matplotlib.pyplot as plt


# ───────────────────────── Helper Functions ──────────────────────────
def _make_metrics(y_true, y_pred, labels_all):
    """Generate standard classification metrics including confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "confusion_matrix": cm,
    }


def _plot_cm(cm, labels, title):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    ax.set(
        title=title,
        xlabel="Predicted",
        ylabel="True",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def QDA_pipeline(
    data: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.20,
    random_state: int = 42,
    use_pca: bool = True,
    pca_var: float = 0.95,
    plot: bool = False,
):
    """
    Train a tuned Quadratic Discriminant Analysis classifier.

    Parameters
    ----------
    data        : DataFrame containing predictors + target
    target_col  : column name for the response variable
    test_size   : proportion of the dataset held out for testing
    random_state: RNG seed for reproducible splits
    use_pca     : whether to apply PCA after scaling (helps collinearity)
    pca_var     : variance to retain if PCA is used
    plot        : if True, show a confusion-matrix heat-map

    Returns
    -------
    dict
        {
          'qda_best'   : fitted QDA pipeline,
          'metrics_qda': metrics dictionary,
          'y_test'     : ground-truth labels,
          'y_pred_qda' : QDA predictions,
        }
    """

    # ───────────── 1 ▸ train / test split ─────────────
    X, y = data.drop(columns=[target_col]), data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # ───────────── 2 ▸ preprocessing ─────────────
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_steps = [("scaler", StandardScaler())]
    if use_pca:
        num_steps.append(("pca", PCA(n_components=pca_var, svd_solver="full")))
    numeric_pipe = Pipeline(num_steps)

    categorical_pipe = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocess = ColumnTransformer(
        [
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    full_pipe = Pipeline(
        [("prep", preprocess), ("varth", VarianceThreshold(threshold=0.0))]
    )

    # ───────────── 3 ▸ QDA hyper-parameter search ─────────────
    qda_pipe = Pipeline(
        [("prep", full_pipe), ("clf", QuadraticDiscriminantAnalysis())]
    )

    param_grid_qda = {"clf__reg_param": [0.0, 0.001]}

    folds = min(5, y_train.value_counts().min())
    cv_scheme = StratifiedKFold(
        n_splits=max(2, folds), shuffle=True, random_state=random_state
    )

    qda_cv = GridSearchCV(
        qda_pipe,
        param_grid=param_grid_qda,
        scoring="balanced_accuracy",
        cv=cv_scheme,
        n_jobs=-1,
        error_score="raise",
    ).fit(X_train, y_train)

    print("✅ Best QDA params:", qda_cv.best_params_)

    y_pred_qda = qda_cv.predict(X_test)

    # ───────────── 4 ▸ metrics ─────────────
    labels_all = np.sort(y.unique())              # fixed order for CM
    metrics_qda = _make_metrics(y_test, y_pred_qda, labels_all)

    # ───────────── 5 ▸ optional plot ─────────────
    if plot:
        _plot_cm(metrics_qda["confusion_matrix"], labels_all, "QDA Confusion Matrix")

    # ───────────── 6 ▸ return bundle ─────────────
    return {
        "qda_best":   qda_cv.best_estimator_,
        "metrics_qda": metrics_qda,
        "y_test":      y_test.reset_index(drop=True),
        "y_pred_qda":  y_pred_qda,
    }



def LogReg_pipeline(
    data: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.20,
    random_state: int = 42,
    use_pca: bool = True,
    pca_var: float = 0.95,
    plot: bool = False,
):
    """
    Hyper-parameter-tuned Logistic‐Regression pipeline.

    Steps
    -----
    1. Split train / test (stratified).
    2. Numeric → StandardScaler [+ optional PCA]; categorical → OneHotEncoder.
    3. VarianceThreshold to drop zero-variance columns.
    4. Grid-search over solver / penalty combos that all support multi-core
       training (`lbfgs` or `saga`) – no `liblinear`, so no n_jobs warning.
    5. Return best model, full metric dict, predictions & ground truth.
    6. Optional confusion-matrix heat-map.

    Returns
    -------
    dict with keys
        'logreg_best', 'metrics_lr', 'y_test', 'y_pred_lr'
    """

    # Silence the inevitable “max_iter reached” chatter during grid search
    warnings.filterwarnings(
        "ignore",
        message="The max_iter was reached which means the coef_ did not converge",
        category=UserWarning,
    )

    # ─── 1 ◂ split ──────────────────────────────────────────────────────────
    X, y = data.drop(columns=[target_col]), data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # ─── 2 ◂ preprocessing blocks ──────────────────────────────────────────
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_steps = [("scaler", StandardScaler())]
    if use_pca:
        num_steps.append(("pca", PCA(n_components=pca_var, svd_solver="full")))
    numeric_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocess = ColumnTransformer(
        [
            ("num", numeric_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    full_pipe = Pipeline(
        [("prep", preprocess), ("varth", VarianceThreshold(threshold=0.0))]
    )

    # ─── 3 ◂ Logistic-Regression grid search ───────────────────────────────
    # baseline estimator – parameters overridden by GridSearchCV
    lr_base = LogisticRegression(max_iter=3000)

    lr_pipe = Pipeline([("prep", full_pipe), ("clf", lr_base)])

    param_grid_lr = [
    {
        "clf__solver": ["saga"],
        "clf__penalty": ["elasticnet"],
        "clf__C": [0.1, 1, 10],
        "clf__l1_ratio": [0.2, 0.5, 0.8],
        "clf__class_weight": ["balanced", None],
        "clf__n_jobs": [-1],
        "clf__max_iter": [5000, 10000],   # ← NEW
        "clf__tol": [1e-4, 1e-3],         # ← NEW (optional)
    },
    {
        "clf__solver": ["saga"],
        "clf__penalty": ["l1", "l2"],
        "clf__C": [0.1, 1, 10],
        "clf__class_weight": ["balanced", None],
        "clf__n_jobs": [-1],
        "clf__max_iter": [5000, 10000],
        "clf__tol": [1e-4, 1e-3],
    },
    {
        "clf__solver": ["lbfgs"],
        "clf__penalty": ["l2"],
        "clf__C": [0.1, 1, 10],
        "clf__class_weight": ["balanced", None],
        "clf__max_iter": [5000, 10000],
        "clf__tol": [1e-4, 1e-3],
    },
]

    folds = min(5, y_train.value_counts().min())
    cv_scheme = StratifiedKFold(
        n_splits=max(2, folds), shuffle=True, random_state=random_state
    )

    lr_cv = GridSearchCV(
        lr_pipe,
        param_grid=param_grid_lr,
        scoring="balanced_accuracy",
        cv=cv_scheme,
        n_jobs=-1,
        error_score="raise",
    ).fit(X_train, y_train)

    print("✅ Best Logistic-Regression params:", lr_cv.best_params_)
    y_pred_lr = lr_cv.predict(X_test)

    # ─── 4 ◂ metrics ───────────────────────────────────────────────────────
    labels_all = np.sort(y.unique())
    metrics_lr = _make_metrics(y_test, y_pred_lr, labels_all)

    # ─── 5 ◂ optional confusion-matrix plot ────────────────────────────────
    if plot:
        _plot_cm(metrics_lr["confusion_matrix"], labels_all, "Logistic Regression — Confusion Matrix")

    # ─── 6 ◂ return bundle ────────────────────────────────────────────────
    return {
        "logreg_best": lr_cv.best_estimator_,
        "metrics_lr": metrics_lr,
        "y_test": y_test.reset_index(drop=True),
        "y_pred_lr": y_pred_lr,
    }



# ─────────────────────────── Random-Forest pipeline ────────────────────────
def RF_pipeline(
    data: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.20,
    random_state: int = 42,
    plot: bool = False,
):
    """Random-Forest with one-hot encoding + hyper-parameter search."""
    X, y = data.drop(columns=[target_col]), data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocess = ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        verbose_feature_names_out=False,
    )

    full_pipe = Pipeline(
        [("prep", preprocess), ("varth", VarianceThreshold(threshold=0.0))]
    )

    rf_pipe = Pipeline(
        [("prep", full_pipe), ("clf", RandomForestClassifier(random_state=random_state))]
    )

    param_grid_rf = {
        "clf__n_estimators": [400, 540],
        "clf__max_depth": [25, 27],
        "clf__min_samples_split": [2, 3],
        "clf__min_samples_leaf": [1, 2],
        "clf__class_weight": [ "balanced"],
    }

    folds = min(5, y_train.value_counts().min())
    cv = StratifiedKFold(n_splits=max(2, folds), shuffle=True, random_state=random_state)

    rf_cv = GridSearchCV(
        rf_pipe,
        param_grid=param_grid_rf,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    ).fit(X_train, y_train)

    print("✅ Best RF params:", rf_cv.best_params_)
    y_pred_rf = rf_cv.predict(X_test)

    labels_all = np.sort(y.unique())
    metrics_rf = _make_metrics(y_test, y_pred_rf, labels_all)

    if plot:
        _plot_cm(metrics_rf["confusion_matrix"], labels_all, "Random Forest")

    return {
        "rf_best": rf_cv.best_estimator_,
        "metrics_rf": metrics_rf,
        "y_test": y_test.reset_index(drop=True),
        "y_pred_rf": y_pred_rf,
    }

# ──────────── XGBoost pipeline (label-safe) ───────────────────
def XGB_pipeline(
    data: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.20,
    random_state: int = 42,
    plot: bool = False,
):
    """
    XGBoost with automatic label-encoding so class values need not be {0,1,…}.
    Returns { 'xgb_best', 'metrics_xgb', 'y_test', 'y_pred_xgb' }.
    """

    warnings.filterwarnings("ignore", category=UserWarning, message=".*will use label encoder")

    # 1 ▸ split --------------------------------------------------------------
    X, y_raw = data.drop(columns=[target_col]), data[target_col]

    le = LabelEncoder().fit(y_raw)          # maps e.g. {1,2} → {0,1}
    y      = pd.Series(le.transform(y_raw), index=y_raw.index)
    labels_all_enc = np.sort(y.unique())    # 0 … k−1
    labels_all_orig = le.inverse_transform(labels_all_enc)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # 2 ▸ preprocessing ------------------------------------------------------
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocess = ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        verbose_feature_names_out=False,
    )

    full_pipe = Pipeline(
        [("prep", preprocess), ("varth", VarianceThreshold(threshold=0.0))]
    )

    # 3 ▸ XGB base estimator -------------------------------------------------
    objective = "binary:logistic" if len(labels_all_enc) == 2 else "multi:softprob"
    xgb_base  = XGBClassifier(
        objective=objective,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    xgb_pipe = Pipeline([("prep", full_pipe), ("clf", xgb_base)])

    # 4 ▸ hyper-parameter grid ----------------------------------------------
    param_grid_xgb = {
        "clf__n_estimators":      [400, 600],
        "clf__learning_rate":     [0.01, 0.05],
        "clf__max_depth":         [4, 6, 7],
        "clf__subsample":         [ 1.0],
        "clf__colsample_bytree":  [0.6,0.8],
    }

    # add class-imbalance handle only for binary
    if len(labels_all_enc) == 2:
        ratio = (y == 0).sum() / (y == 1).sum()
        param_grid_xgb["clf__scale_pos_weight"] = [1, ratio]

    folds = min(5, y_train.value_counts().min())
    cv = StratifiedKFold(
        n_splits=max(2, folds), shuffle=True, random_state=random_state
    )

    xgb_cv = GridSearchCV(
        xgb_pipe,
        param_grid=param_grid_xgb,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    ).fit(X_train, y_train)

    print("✅ Best XGB params:", xgb_cv.best_params_)
    y_pred_enc = xgb_cv.predict(X_test)
    y_pred_orig = le.inverse_transform(y_pred_enc)
    y_test_orig = le.inverse_transform(y_test)

    # 5 ▸ metrics ------------------------------------------------------------
    metrics_xgb = _make_metrics(y_test_orig, y_pred_orig, labels_all_orig)

    # 6 ▸ plot ---------------------------------------------------------------
    if plot:
        _plot_cm(metrics_xgb["confusion_matrix"], labels_all_orig, "XGBoost")

    # 7 ▸ return -------------------------------------------------------------
    return {
        "xgb_best":   xgb_cv.best_estimator_,
        "metrics_xgb": metrics_xgb,
        "y_test":      pd.Series(y_test_orig, index=X_test.index),
        "y_pred_xgb":  pd.Series(y_pred_orig, index=X_test.index),
    }



class StackedEnsembler(BaseEstimator, ClassifierMixin):
    """
    Simple stacking wrapper that:
      1. Re-fits each base model with K-fold CV to create out-of-fold (OOF) meta-features
      2. Trains a meta-learner on those OOF features
      3. Exposes predict / predict_proba that pipe new data through the stack

    Parameters
    ----------
    base_models : Sequence[Tuple[str, BaseEstimator]]
        List of (name, estimator) pairs.  Each estimator **must** implement
        fit / predict_proba following sklearn API.
    meta_model : BaseEstimator | str
        The meta-learner.  Pass a pre-instantiated estimator *or*
        `"logistic"` / `"xgb"` / `"ridge"` to get a sensible default.
    cv : int
        Number of folds used to generate OOF predictions.
    random_state : int | None
        Random seed for the internal KFold splitter.
    """

    def __init__(
        self,
        base_models: Sequence[Tuple[str, BaseEstimator]],
        meta_model: Union[str, BaseEstimator] = "logistic",
        cv: int = 5,
        random_state: int | None = 42,
    ):
        self.base_models = [(name, clone(model)) for name, model in base_models]
        self.meta_model = self._init_meta(meta_model, random_state)
        self.cv = cv
        self.random_state = random_state

    # ------------------------------------------------------------------ #

    @staticmethod
    def _init_meta(meta_model: Union[str, BaseEstimator], random_state: int | None):
        if isinstance(meta_model, str):
            if meta_model.lower() == "logistic":
                return LogisticRegression(
                    max_iter=1000, penalty="l2", random_state=random_state
                )
            elif meta_model.lower() == "ridge":
                from sklearn.linear_model import RidgeClassifier
                return RidgeClassifier(random_state=random_state)
            elif meta_model.lower() == "xgb":
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    random_state=random_state,
                )
            else:
                raise ValueError(f"Unknown meta_model string '{meta_model}'.")
        else:
            return clone(meta_model)

    # ------------------------------------------------------------------ #

    def fit(self, X, y):
        """Fit base models via K-fold OOF and then fit the meta-learner."""
        kf = KFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        n_samples = len(X)
        n_models = len(self.base_models)
        self._oof_preds_ = np.zeros((n_samples, n_models))

        # Build meta-features
        for m_idx, (name, model) in enumerate(self.base_models):
            for train_idx, val_idx in kf.split(X):
                # Handle both DataFrame and numpy array inputs
                if hasattr(X, 'iloc'):
                    X_train_fold = X.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                else:
                    X_train_fold = X[train_idx]
                    X_val_fold = X[val_idx]
                
                # Handle both Series and numpy array inputs for y
                if hasattr(y, 'iloc'):
                    y_train_fold = y.iloc[train_idx]
                else:
                    y_train_fold = y[train_idx]
                
                model_fold = clone(model).fit(X_train_fold, y_train_fold)
                self._oof_preds_[val_idx, m_idx] = model_fold.predict_proba(
                    X_val_fold
                )[:, 1]

            # Fit the model on the full data for inference later
            model.fit(X, y)

        # Fit meta-model
        self.meta_model.fit(self._oof_preds_, y)
        return self

    # ------------------------------------------------------------------ #

    def _meta_features(self, X) -> np.ndarray:
        """Generate stacked features for new data."""
        return np.column_stack(
            [model.predict_proba(X)[:, 1] for _, model in self.base_models]
        )

    # sklearn API ------------------------------------------------------- #

    def predict_proba(self, X) -> np.ndarray:
        meta_X = self._meta_features(X)
        return self.meta_model.predict_proba(meta_X)

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def Stacked_pipeline(
    data: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.20,
    random_state: int = 42,
    use_pca: bool = True,
    pca_var: float = 0.95,
    meta_model: Union[str, BaseEstimator] = "logistic",
    cv_folds: int = 5,
    plot: bool = False,
):

    
    # ───────────── 1 ▸ train / test split ─────────────
    X, y = data.drop(columns=[target_col]), data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    
    # ───────────── 2 ▸ preprocessing setup ─────────────
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Preprocessing for models that need scaling/PCA (QDA, LogReg)
    num_steps = [("scaler", StandardScaler())]
    if use_pca:
        num_steps.append(("pca", PCA(n_components=pca_var, svd_solver="full")))
    numeric_pipe_scaled = Pipeline(num_steps)
    
    categorical_pipe = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    
    preprocess_scaled = ColumnTransformer(
        [
            ("num", numeric_pipe_scaled, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    
    # Preprocessing for tree-based models (RF, XGB) - no scaling needed
    preprocess_trees = ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        verbose_feature_names_out=False,
    )
    
    full_pipe_scaled = Pipeline(
        [("prep", preprocess_scaled), ("varth", VarianceThreshold(threshold=0.0))]
    )
    
    full_pipe_trees = Pipeline(
        [("prep", preprocess_trees), ("varth", VarianceThreshold(threshold=0.0))]
    )
    
    # ───────────── 3 ▸ define base models ─────────────
    
    # QDA model
    qda_model = Pipeline([
        ("prep", full_pipe_scaled),
        ("clf", QuadraticDiscriminantAnalysis(reg_param=0.001))
    ])
    
    # Logistic Regression model
    logreg_model = Pipeline([
        ("prep", full_pipe_scaled),
        ("clf", LogisticRegression(
            max_iter=5000,
            solver="saga",
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    
    # Random Forest model
    rf_model = Pipeline([
        ("prep", full_pipe_trees),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    
    # XGBoost model (with label encoding for compatibility)
    le = LabelEncoder().fit(y_train)
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    objective = "binary:logistic" if len(np.unique(y_train_encoded)) == 2 else "multi:softprob"
    xgb_model = Pipeline([
        ("prep", full_pipe_trees),
        ("clf", XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=1.0,
            colsample_bytree=0.8,
            objective=objective,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    
    # ───────────── 4 ▸ create base models list ─────────────
    base_models = [
        ("qda", qda_model),
        ("logreg", logreg_model),
        ("rf", rf_model),
        ("xgb", xgb_model),
    ]
    
    # ───────────── 5 ▸ train individual models for comparison ─────────────
    base_models_performance = {}
    labels_all = np.sort(y.unique())
    
    for name, model in base_models:
        print(f"Training {name.upper()} base model...")
        if name == "xgb":
            # XGBoost needs encoded labels
            model.fit(X_train, y_train_encoded)
            y_pred = le.inverse_transform(model.predict(X_test))
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        base_models_performance[name] = _make_metrics(y_test, y_pred, labels_all)
        print(f"✅ {name.upper()} - Balanced Accuracy: {base_models_performance[name]['balanced_accuracy']:.4f}")
    
    # ───────────── 6 ▸ create and train stacked ensemble ─────────────
    print(f"\nTraining Stacked Ensemble with {meta_model} meta-learner...")
    
    # For stacking, we need to use encoded labels if any base model requires it
    stacked_ensemble = StackedEnsembler(
        base_models=base_models,
        meta_model=meta_model,
        cv=cv_folds,
        random_state=random_state
    )
    
    # Train the stacked ensemble
    if any(name == "xgb" for name, _ in base_models):
        # Use encoded labels for training
        stacked_ensemble.fit(X_train, y_train_encoded)
        y_pred_stacked_encoded = stacked_ensemble.predict(X_test)
        y_pred_stacked = le.inverse_transform(y_pred_stacked_encoded)
    else:
        stacked_ensemble.fit(X_train, y_train)
        y_pred_stacked = stacked_ensemble.predict(X_test)
    
    # ───────────── 7 ▸ evaluate stacked ensemble ─────────────
    metrics_stacked = _make_metrics(y_test, y_pred_stacked, labels_all)
    
    print(f"✅ STACKED ENSEMBLE - Balanced Accuracy: {metrics_stacked['balanced_accuracy']:.4f}")
    
    # ───────────── 8 ▸ optional plot ─────────────
    if plot:
        _plot_cm(metrics_stacked["confusion_matrix"], labels_all, "Stacked Ensemble — Confusion Matrix")
    
    # ───────────── 9 ▸ return results ─────────────
    return {
        "stacked_best": stacked_ensemble,
        "metrics_stacked": metrics_stacked,
        "y_test": y_test.reset_index(drop=True),
        "y_pred_stacked": y_pred_stacked,
        "base_models_performance": base_models_performance,
    }