from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np



class FeatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_originals: bool = True):
        
        self.drop_originals = drop_originals
    
    # questa funzione fa sì che sklearn NON converta in ndarray
    def _more_tags(self):
        return {"preserves_dataframe": True}

    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        X = X.copy()

        # Average grade across semesters
        X["weighted_avg_grade"] = np.where(
            (X["Curricular_units_1st_sem_approved"] + X["Curricular_units_2nd_sem_approved"]) > 0,
            (X["Curricular_units_1st_sem_grade"] * X["Curricular_units_1st_sem_approved"] + X["Curricular_units_2nd_sem_grade"] * X["Curricular_units_2nd_sem_approved"]) /
            (X["Curricular_units_1st_sem_approved"] + X["Curricular_units_2nd_sem_approved"]),
            0.0 
        )

        # Pass rate per semester (avoid division by zero)
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

        X["approved_delta"] = (
            X["Curricular_units_2nd_sem_approved"]
            - X["Curricular_units_1st_sem_approved"]
        )

        X["total_enrolled"] = (
            X["Curricular_units_1st_sem_enrolled"]
            + X["Curricular_units_2nd_sem_enrolled"]
        )
        
        # Age bins
        X['age_bin'] = pd.cut(X['Age'], bins=[0, 20, 25, 100], labels=['young', 'medium', 'adult'])

        # Parental background score 
        def background_score(row):
            score = 0
            if str(row['Mother_qualification']).isdigit():
                score += int(row['Mother_qualification'])
            if str(row['Father_qualification']).isdigit():
                score += int(row['Father_qualification'])
            return score

        X['parent_background_score'] = X.apply(background_score, axis=1)

        # Drop original columns now represented by engineered feature
        X.drop(columns=[
            'Mother_qualification',
            'Father_qualification'
        ], inplace=True)

        return X
    
