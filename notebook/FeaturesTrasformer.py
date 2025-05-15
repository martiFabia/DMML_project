from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that creates new aggregate and delta features
    from semester-based curricular data.
    """
    def __init__(self, drop_originals: bool = True):
        """
        Parameters
        ----------
        drop_originals : bool
            Whether to drop the raw semester columns after creating
            the new features.
        """
        self.drop_originals = drop_originals
    
    # 👇  questa funzione fa sì che sklearn NON converta in ndarray
    def _more_tags(self):
        return {"preserves_dataframe": True}

    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X : pd.DataFrame or numpy array
            Input data with columns for first‐ and second‐semester metrics.
        y : Ignored

        Returns
        -------
        np.ndarray
            Transformed array with new features (and without
            the dropped originals if drop_originals=True).
        """
        # Ensure we work with a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        X = X.copy()

        # 1) Average grade across semesters
        X["avg_grade"] = X[
            ["Curricular_units_1st_sem_grade", "Curricular_units_2nd_sem_grade"]
        ].mean(axis=1)

        # 2) Pass rate per semester (avoid division by zero)
        X["pass_rate_1st"] = np.where(
            X["Curricular_units_1st_sem_enrolled"] > 0,
            X["Curricular_units_1st_sem_approved"]
            / X["Curricular_units_1st_sem_enrolled"],
            0.0
        )
        X["pass_rate_2nd"] = np.where(
            X["Curricular_units_2nd_sem_enrolled"] > 0,
            X["Curricular_units_2nd_sem_approved"]
            / X["Curricular_units_2nd_sem_enrolled"],
            0.0
        )

        # 3) Change in pass rate between semesters
        X["pass_rate_delta"] = (
            X["pass_rate_2nd"] - X["pass_rate_1st"]
        )

        # 4) Grade and approved‐units deltas
        X["grade_delta"] = (
            X["Curricular_units_2nd_sem_grade"]
            - X["Curricular_units_1st_sem_grade"]
        )
        X["approved_delta"] = (
            X["Curricular_units_2nd_sem_approved"]
            - X["Curricular_units_1st_sem_approved"]
        )

        # 5) Total workload aggregates
        X["total_enrolled"] = (
            X["Curricular_units_1st_sem_enrolled"]
            + X["Curricular_units_2nd_sem_enrolled"]
        )
        X["total_credited"] = (
            X["Curricular_units_1st_sem_credited"]
            + X["Curricular_units_2nd_sem_credited"]
        )

        # 6) Evaluation completion rate per semester
        #    clip to avoid division by zero
        first_evals = (
            X["Curricular_units_1st_sem_evaluations"]
            + X["Curricular_units_1st_sem_without_evaluations"]
        ).clip(lower=1)

        X["eval_completion_rate_1st"] = (
            X["Curricular_units_1st_sem_evaluations"] / first_evals
        )

        second_evals = (
            X["Curricular_units_2nd_sem_evaluations"]
            + X["Curricular_units_2nd_sem_without_evaluations"]
        ).clip(lower=1)

        X["eval_completion_rate_2nd"] = (
            X["Curricular_units_2nd_sem_evaluations"] / second_evals
        )
        X["eval_completion_delta"] = (
            X["eval_completion_rate_2nd"]
            - X["eval_completion_rate_1st"]
        )

        # 7)Drop the raw features
        # if self.drop_originals:
        raw_cols = [
            'Curricular_units_1st_sem_credited',
            'Curricular_units_1st_sem_enrolled',
            'Curricular_units_1st_sem_evaluations',
            'Curricular_units_1st_sem_approved',
            'Curricular_units_1st_sem_grade',
            'Curricular_units_1st_sem_without_evaluations',
            'Curricular_units_2nd_sem_credited',
            'Curricular_units_2nd_sem_enrolled',
            'Curricular_units_2nd_sem_evaluations',
            'Curricular_units_2nd_sem_approved',
            'Curricular_units_2nd_sem_grade',
            'Curricular_units_2nd_sem_without_evaluations'
        ]
        X = X.drop(columns=[c for c in raw_cols if c in X.columns])

        # X=X.drop(columns=['Curricular_units_1st_sem_credited',
        #                     'Curricular_units_1st_sem_enrolled',
        #                     'Curricular_units_1st_sem_evaluations',
        #                     'Curricular_units_1st_sem_approved',
        #                     'Curricular_units_1st_sem_grade',
        #                     'Curricular_units_1st_sem_without_evaluations',
        #                     'Curricular_units_2nd_sem_credited',
        #                     'Curricular_units_2nd_sem_enrolled',
        #                     'Curricular_units_2nd_sem_evaluations',
        #                     'Curricular_units_2nd_sem_approved',
        #                     'Curricular_units_2nd_sem_grade',
        #                     'Curricular_units_2nd_sem_without_evaluations'])

     
        return X
