from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer




cat_cols = ["Application_mode", "Course", "Gender", "Previous_qualification",  
            "Application_order",  "age_bin", "Mother_occupation",
            "Father_occupation"]   

num_cols = [ "Previous_qualification_grade", "Admission_grade", "Age", "Unemployment_rate", 
             "GDP", "weighted_avg_grade", "parent_background_score",
             'Curricular_units_1st_sem_credited', 'Curricular_units_2nd_sem_credited', 
             'Curricular_units_1st_sem_enrolled', 'Curricular_units_2nd_sem_enrolled',
             'Curricular_units_1st_sem_evaluations', 'Curricular_units_2nd_sem_evaluations',
             'Curricular_units_1st_sem_approved', 'Curricular_units_2nd_sem_approved',
             'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_without_evaluations',
             'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade',
             'pass_rate_1st', 'pass_rate_2nd', 'approved_delta', 'total_enrolled']

binary_cols = ["Daytime/evening_attendance","Displaced", "Debtor", "Tuition_fees_up_to_date", "Scholarship_holder"]



# Preprocessing pipeline for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('bin', "passthrough", binary_cols),
        ('oh', OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ])



