# feature_selection_experiments.py
# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    RFE,
    SequentialFeatureSelector,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier  # type: ignore
from lightgbm import LGBMClassifier  # type: ignore


from FeaturesTrasformer import FeaturesTransformer 
from preprocessor import preprocessor 

# Silence unhelpful warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Model registry
# =============================================================================
MODELS: Dict[str, BaseEstimator] = {
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Logistic Regression': LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, class_weight='balanced'),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'ExtraTrees': ExtraTreesClassifier(random_state=42, class_weight='balanced'),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Bagging': BaggingClassifier(random_state=42),
    'XGBoost': XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', random_state=42),
    'LightGBM': LGBMClassifier(objective='multiclass', num_class=3, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# =============================================================================
# Custom CFS (Correlation‑based Feature Selection) implementation
# =============================================================================
class CFSSelector(BaseEstimator, TransformerMixin):
    """A very simple CFS filter using absolute Pearson correlation.

    WARNING: This implementation assumes *dense*, numerical features. For one‑hot
    encoded data it still works (0/1 values) but may not be optimal. Replace
    with a library implementation (e.g. `skrebate`, `itmo‑fs`) for production.
    """

    def __init__(self, k: int):
        self.k = k
        self.support_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):  # type: ignore[override]

        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        y = np.asarray(y).ravel()
        # Compute absolute Pearson correlation between each feature and target
        corrs = np.abs(np.corrcoef(X.T, y)[-1, :-1])
        # Rank features by correlation *descending*
        top_idx = np.argsort(corrs)[::-1][: self.k]
        self.support_ = np.zeros_like(corrs, dtype=bool)
        self.support_[top_idx] = True
        return self

    def transform(self, X: np.ndarray):  # type: ignore[override]
        if self.support_ is None:
            raise ValueError("CFSSelector has not been fitted yet.")
        return X[:, self.support_]

    def get_support(self) -> np.ndarray:  # for compatibility with SelectKBest
        if self.support_ is None:
            raise ValueError("CFSSelector has not been fitted yet.")
        return self.support_

# =============================================================================
# Feature‑selector builders (return a fresh selector for a given k)
# =============================================================================
SelectorBuilder = Callable[[int], BaseEstimator]

SELECTOR_BUILDERS: Dict[str, SelectorBuilder] = {
    # K‑Best filters
    "KBest‑MI": lambda k: SelectKBest(score_func=mutual_info_classif, k=k),
    "KBest‑CHI2": lambda k: SelectKBest(score_func=chi2, k=k),
    "KBest‑ANOVA": lambda k: SelectKBest(score_func=f_classif, k=k),
    # Correlation‑based filter
    "CFS": lambda k: CFSSelector(k=k),
    # Wrapper methods (RFE & SFS use RF by default; adjust if needed)
    "RFE": lambda k: RFE(
        estimator=RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        n_features_to_select=k,
        step=0.2,
    ),
    "SFS": lambda k: SequentialFeatureSelector(
        estimator=RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        n_features_to_select=k,
        direction="forward",
        n_jobs=-1,
    ),
}

# =============================================================================
# Pipeline factory
# =============================================================================

def build_pipeline(model: BaseEstimator, selector: BaseEstimator) -> Pipeline:
    """Return a full pipeline with preprocessing, selection, SMOTE and model."""
    return Pipeline(
        steps=[
            ("feature_transformer", FeaturesTransformer(drop_originals=True)),
            ("preprocessing", preprocessor),
            ("selector", selector),
            ("smote", SMOTE(random_state=42)),
            ("classifier", model),
        ]
    )

# =============================================================================
# Core experiment runner
# =============================================================================

Scoring: Dict[str, str | Callable] = {
    "f1_macro": make_scorer(f1_score, average="macro"),
    "balanced_accuracy": "balanced_accuracy",
}


def run_feature_selection_experiments(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    models: Dict[str, BaseEstimator],
    selector_builders: Dict[str, SelectorBuilder],
    k_values: Sequence[int],
    cv_splits: int = 5,
    results_path: str | Path = "feature_selection_results.txt",
) -> None:
    """Loop over (model × selector × k) and append metrics to a txt file."""
    path = Path(results_path)
    # Header – overwrite existing file
    path.write_text(
        "Model\tSelector\tK\tF1_macro_mean\tF1_macro_std\tBalAcc_mean\tBalAcc_std\n",
        encoding="utf‑8",
    )

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for model_name, model in models.items():
        for selector_name, selector_builder in selector_builders.items():
            for k in k_values:
                selector = selector_builder(k)
                pipe = build_pipeline(model, selector)

                cv_res = cross_validate(
                    pipe,
                    X,
                    y,
                    cv=cv,
                    scoring=Scoring,
                    n_jobs=-1,
                    error_score="raise",
                )

                f1_mean = cv_res["test_f1_macro"].mean()
                f1_std = cv_res["test_f1_macro"].std()
                bal_mean = cv_res["test_balanced_accuracy"].mean()
                bal_std = cv_res["test_balanced_accuracy"].std()

                # Append a line to the results file
                line = (
                    f"{model_name}\t{selector_name}\t{k}\t"
                    f"{f1_mean:.4f}\t{f1_std:.4f}\t{bal_mean:.4f}\t{bal_std:.4f}\n"
                )
                with path.open("a", encoding="utf‑8") as f:
                    f.write(line)

                print(line.strip())  # real‑time feedback in console


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported. See docstring for example usage."
    )
