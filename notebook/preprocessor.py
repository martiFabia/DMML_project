from sklearn.preprocessing import OneHotEncoder
from category_encoders.target_encoder import TargetEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

cat_cols = ["Marital_status", "Application_mode", "Course", "Previous_qualification", "Mother_occupation",
               "Father_qualification", "Mother_qualification", "Father_occupation"]   

num_cols = ["Application_order", "Previous_qualification_grade", 
            "Admission_grade", "Age", "Unemployment_rate", "Inflation_rate", "GDP", "avg_grade", "pass_rate_1st", 
            "pass_rate_2nd", "pass_rate_delta", "grade_delta",
            "approved_delta", "total_enrolled", "total_credited", "eval_completion_rate_1st", 
            "eval_completion_rate_2nd", "eval_completion_delta"]

binary_cols = ["Daytime/evening_attendance","Displaced", "Debtor", "Tuition_fees_up_to_date", "Gender", "Scholarship_holder"]

# Preprocessing pipeline for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('bin', "passthrough", binary_cols),
        ('oh', OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ])