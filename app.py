import tkinter as tk
from tkinter import ttk
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import warnings
sys.path.append(os.path.join(os.path.dirname(__file__), 'notebook'))

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# Categorical features
categorical_features = {
    'Application_mode': [
        "1 - 1st phase - general contingent",
        "2 - Ordinance No. 612/93",
        "5 - 1st phase - special contingent (Azores Island)",
        "7 - Holders of other higher courses",
        "10 - Ordinance No. 854-B/99",
        "15 - International student (bachelor)",
        "16 - 1st phase - special contingent (Madeira Island)",
        "17 - 2nd phase - general contingent",
        "18 - 3rd phase - general contingent",
        "26 - Ordinance No. 533-A/99, item b2) (Different Plan)",
        "27 - Ordinance No. 533-A/99, item b3 (Other Institution)",
        "39 - Over 23 years old",
        "42 - Transfer",
        "43 - Change of course",
        "44 - Technological specialization diploma holders",
        "51 - Change of institution/course",
        "53 - Short cycle diploma holders",
        "57 - Change of institution/course (International)"
    ],
    'Course': [
        "33 - Biofuel Production Technologies",
        "171 - Animation and Multimedia Design",
        "8014 - Social Service (evening attendance)",
        "9003 - Agronomy",
        "9070 - Communication Design",
        "9085 - Veterinary Nursing",
        "9119 - Informatics Engineering",
        "9130 - Equinculture",
        "9147 - Management",
        "9238 - Social Service",
        "9254 - Tourism",
        "9500 - Nursing",
        "9556 - Oral Hygiene",
        "9670 - Advertising and Marketing Management",
        "9773 - Journalism and Communication",
        "9853 - Basic Education",
        "9991 - Management (evening attendance)"
    ],
    'Previous_qualification': [
        "1 - Secondary education",
        "2 - Higher education - bachelor's degree",
        "3 - Higher education - degree",
        "4 - Higher education - master's",
        "5 - Higher education - doctorate",
        "6 - Frequency of higher education",
        "9 - 12th year of schooling - not completed",
        "10 - 11th year of schooling - not completed",
        "12 - Other - 11th year of schooling",
        "14 - 10th year of schooling",
        "15 - 10th year of schooling - not completed",
        "19 - Basic education 3rd cycle (9th/10th/11th year) or equiv.",
        "38 - Basic education 2nd cycle (6th/7th/8th year) or equiv.",
        "39 - Technological specialization course",
        "40 - Higher education - degree (1st cycle)",
        "42 - Professional higher technical course",
        "43 - Higher education - master (2nd cycle)"
    ],
    'Mother_qualification': [
        "1 - Secondary Education - 12th Year of Schooling or Eq.",
        "2 - Higher Education - Bachelor's Degree",
        "3 - Higher Education - Degree",
        "4 - Higher Education - Master's",
        "5 - Higher Education - Doctorate",
        "6 - Frequency of Higher Education",
        "9 - 12th Year of Schooling - Not Completed",
        "10 - 11th Year of Schooling - Not Completed",
        "11 - 7th Year (Old)",
        "12 - Other - 11th Year of Schooling",
        "14 - 10th Year of Schooling",
        "18 - General commerce course",
        "19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
        "22 - Technical-professional course",
        "26 - 7th year of schooling",
        "27 - 2nd cycle of the general high school course",
        "29 - 9th Year of Schooling - Not Completed",
        "30 - 8th year of schooling",
        "34 - Unknown",
        "35 - Can't read or write",
        "36 - Can read without having a 4th year of schooling",
        "37 - Basic education 1st cycle (4th/5th year) or equiv.",
        "38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
        "39 - Technological specialization course",
        "40 - Higher education - degree (1st cycle)",
        "41 - Specialized higher studies course",
        "42 - Professional higher technical course",
        "43 - Higher Education - Master (2nd cycle)",
        "44 - Higher Education - Doctorate (3rd cycle)"
    ],
    'Father_qualification': [
        "1 - Secondary Education - 12th Year of Schooling or Eq.",
        "2 - Higher Education - Bachelor's Degree",
        "3 - Higher Education - Degree",
        "4 - Higher Education - Master's",
        "5 - Higher Education - Doctorate",
        "6 - Frequency of Higher Education",
        "9 - 12th Year of Schooling - Not Completed",
        "10 - 11th Year of Schooling - Not Completed",
        "11 - 7th Year (Old)",
        "12 - Other - 11th Year of Schooling",
        "13 - 2nd year complementary high school course",
        "14 - 10th Year of Schooling",
        "18 - General commerce course",
        "19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
        "20 - Complementary High School Course",
        "22 - Technical-professional course",
        "25 - Complementary High School Course - not concluded",
        "26 - 7th year of schooling",
        "27 - 2nd cycle of the general high school course",
        "29 - 9th Year of Schooling - Not Completed",
        "30 - 8th year of schooling",
        "31 - General Course of Administration and Commerce",
        "33 - Supplementary Accounting and Administration",
        "34 - Unknown",
        "35 - Can't read or write",
        "36 - Can read without having a 4th year of schooling",
        "37 - Basic education 1st cycle (4th/5th year) or equiv.",
        "38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
        "39 - Technological specialization course",
        "40 - Higher education - degree (1st cycle)",
        "41 - Specialized higher studies course",
        "42 - Professional higher technical course",
        "43 - Higher Education - Master (2nd cycle)",
        "44 - Higher Education - Doctorate (3rd cycle)"
    ],
    'Mother_occupation': [
        "0 - Student", "1 - Legislative/Executive/Directors",
        "2 - Intellectual/Scientific Activities", "3 - Intermediate Level Technicians",
        "4 - Administrative staff", "5 - Services/Sellers/Security",
        "6 - Farmers and Agriculture workers", "7 - Industry/Construction/Craftsmen",
        "8 - Machine Operators", "9 - Unskilled Workers", "10 - Armed Forces",
        "90 - Other Situation", "99 - (blank)", "122 - Health professionals", "123 - Teachers",
        "125 - ICT Specialists", "131 - Engineering/Science Technicians",
        "132 - Health Technicians", "134 - Legal/Social/Sports/Cultural Technicians",
        "141 - Office/Data Processing", "143 - Financial/Registry Operators",
        "144 - Other Admin Staff", "151 - Personal Service Workers", "152 - Sellers",
        "153 - Personal Care Workers", "171 - Skilled Construction Workers",
        "173 - Artisans/Precision Workers", "175 - Food/Wood/Textile Craftsmen",
        "191 - Cleaning Workers", "192 - Unskilled Agriculture/Fishery",
        "193 - Unskilled Construction/Transport", "194 - Meal Prep Assistants"
    ],
    'Father_occupation': [
        "0 - Student", "1 - Legislative/Executive/Directors",
        "2 - Intellectual/Scientific Activities", "3 - Intermediate Level Technicians",
        "4 - Administrative staff", "5 - Services/Sellers/Security",
        "6 - Farmers and Agriculture workers", "7 - Industry/Construction/Craftsmen",
        "8 - Machine Operators", "9 - Unskilled Workers", "10 - Armed Forces",
        "90 - Other Situation", "99 - (blank)", "101 - Armed Forces Officers",
        "102 - Armed Forces Sergeants", "103 - Armed Forces personnel",
        "112 - Admin/Commercial Directors", "114 - Services Directors",
        "121 - Engineering/Math/Science Specialists", "122 - Health professionals",
        "123 - Teachers", "124 - Finance/Admin Specialists",
        "131 - Science/Engineering Technicians", "132 - Health Technicians",
        "134 - Legal/Social/Cultural Technicians", "135 - ICT Technicians",
        "141 - Office/Data Processing", "143 - Financial/Registry Operators",
        "144 - Other Admin Staff", "151 - Personal Service Workers", "152 - Sellers",
        "153 - Personal Care Workers", "154 - Protection/Security Staff",
        "161 - Market-Oriented Farmers", "163 - Subsistence Farmers/Fishers",
        "171 - Skilled Construction Workers", "172 - Metal/Mechanical Workers",
        "174 - Electric/Electronic Workers", "175 - Food/Wood/Textile Craftsmen",
        "181 - Plant/Machine Operators", "182 - Assembly Workers",
        "183 - Vehicle Drivers", "192 - Unskilled Agriculture Workers",
        "193 - Unskilled Construction/Transport", "194 - Meal Prep Assistants",
        "195 - Street Vendors"
    ],
    'Application_order': ['0','1', '2', '3', '4', '5', '6', '7', '8', '9'],

    'Daytime/evening_attendance': ['1 - Daytime', '0 - Evening'],
    'Displaced': ['1 - Yes', '0 - No'],
    'Debtor': ['1 - Yes', '0 - No'],
    'Tuition_fees_up_to_date': ['1 - Yes', '0 - No'],
    'Gender': ['1 - Male', '0 - Female'],
    'Scholarship_holder': ['1 - Yes', '0 - No']
}

# Numerical features
numerical_features = [
    'Previous_qualification_grade', 'Admission_grade', 'Age',
    'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations',
    'Unemployment_rate', 'GDP'
]

label_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

class StudentFormApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Dropout Predictor")

        # --- Frame with scrollbar ---
        container = tk.Frame(root)
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        container.pack(fill="both", expand=True)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Title and description ---
        tk.Label(self.scrollable_frame, text="🎓 Student Outcome Predictor", font=("Helvetica", 18, "bold")).grid(
                row=0, column=0, columnspan=4, pady=(10, 5), sticky="n")

        tk.Label(self.scrollable_frame,
                text="Fill in the student's information below. The app will predict whether the student will Dropout, Enroll, or Graduate.",
                wraplength=500, justify="center", font=("Arial", 10)).grid(
                row=1, column=0, columnspan=4, pady=(0, 15), sticky="n")


        row = 2
        self.inputs = {}

        # Model 
        tk.Label(
            self.scrollable_frame,
            text="Model:  Random Forest",
            font=("Helvetica", 12, "bold")
        ).grid(row=row, column=0, columnspan=4, sticky="n", pady=5)

        
        row += 1

        # Pairing categorical and numerical features side by side
        cat_num_pairs = list(zip(categorical_features.items(), numerical_features))
        for (cat_feat, cat_options), num_feat in cat_num_pairs:
            # Categorical
            tk.Label(self.scrollable_frame, text=cat_feat, anchor="center").grid(row=row, column=0, sticky="e", padx=10, pady=5)
            cat_var = tk.StringVar()
            cat_dropdown = ttk.Combobox(self.scrollable_frame, textvariable=cat_var, values=cat_options, state='readonly')
            cat_dropdown.grid(row=row, column=1, padx=10, pady=5)
            self.inputs[cat_feat] = cat_var

            # Numerical
            tk.Label(self.scrollable_frame, text=num_feat, anchor="center").grid(row=row, column=2, sticky="e", padx=10, pady=5)
            num_entry = tk.Entry(self.scrollable_frame)
            num_entry.grid(row=row, column=3, padx=10, pady=5)
            self.inputs[num_feat] = num_entry

            row += 1

        # Se ci sono feature dispari, gestisci quelle rimanenti
        if len(categorical_features) > len(numerical_features):
            for cat_feat in list(categorical_features.keys())[len(numerical_features):]:
                tk.Label(self.scrollable_frame, text=cat_feat, anchor="center").grid(row=row, column=0, sticky="e", padx=10, pady=5)
                cat_var = tk.StringVar()
                cat_dropdown = ttk.Combobox(self.scrollable_frame, textvariable=cat_var, values=categorical_features[cat_feat], state='readonly')
                cat_dropdown.grid(row=row, column=1, padx=10, pady=5)
                self.inputs[cat_feat] = cat_var
                row += 1

        elif len(numerical_features) > len(categorical_features):
            for num_feat in numerical_features[len(categorical_features):]:
                tk.Label(self.scrollable_frame, text=num_feat, anchor="center").grid(row=row, column=2, sticky="e", padx=10, pady=5)
                num_entry = tk.Entry(self.scrollable_frame)
                num_entry.grid(row=row, column=3, padx=10, pady=5)
                self.inputs[num_feat] = num_entry
                row += 1

        # Predict and reset buttons
        button_frame = tk.Frame(self.scrollable_frame)
        button_frame.grid(row=row, column=0, columnspan=4, pady=10)

        tk.Button(button_frame, text="Predict", command=self.on_predict).pack(side="left", padx=20)
        tk.Button(button_frame, text="Reset", command=self.on_reset).pack(side="left", padx=20)
        row += 1

        # Result label
        self.result_label = tk.Label(self.scrollable_frame, text="", font=("Arial", 12), fg="blue")
        self.result_label.grid(row=row, column=0, columnspan=4, pady=10)
        row += 1
        # SHAP explanation label
        self.shap_label = tk.Label(self.scrollable_frame, text="", font=("Courier", 10))
        self.shap_label.grid(row=row, column=0, columnspan=4, pady=10)

    

    def on_predict(self):
        model_name = 'random_forest'
        model_path = f"models_SMOTE/best_model_{model_name}.joblib"

        if not os.path.exists(model_path):
            self.result_label.config(text=f"Model file not found: {model_path}", fg="red")
            return

        try:
            pipeline = joblib.load(model_path)
        except Exception as e:
            self.result_label.config(text=f"Failed to load model: {e}", fg="red")
            return

        input_data = {}

        # Handle categorical features
        for feature in categorical_features:
            val = self.inputs[feature].get()
            if feature == "Application_order":
                input_data[feature] = [val]
            else:
                try:
                    code = int(val.split(" - ")[0])
                except:
                    self.result_label.config(text=f"Invalid input for {feature}", fg="red")
                    return
                input_data[feature] = [code]

        # Handle numeric features
        for feature in numerical_features:
            try:
                value = float(self.inputs[feature].get())
            except:
                self.result_label.config(text=f"Invalid numeric value for {feature}", fg="red")
                return
            input_data[feature] = [value]

        df = pd.DataFrame(input_data)

        try:
            prediction = pipeline.predict(df)[0]
            if isinstance(prediction, np.ndarray):
                prediction = prediction.item()

            # Gestisci se è int (class index) oppure stringa diretta (es. 'Dropout')
            if isinstance(prediction, str):
                result = prediction
            else:
                result = label_mapping.get(prediction, f"Unknown ({prediction})")
            color = {"Dropout": "red", "Enrolled": "orange", "Graduate": "green"}.get(result, "black")
            self.result_label.config(text=f"Predicted outcome: {result}", fg=color)

            explanation_text = self.explain_prediction(pipeline, df)
            prob_array = pipeline.predict_proba(df)[0]  # es: [0.2, 0.3, 0.5]
            class_labels = ["Dropout", "Enrolled", "Graduate"]
            probs = dict(zip(class_labels, prob_array))

            self.show_prediction_details(model_name=model_name, probs=probs, shap_explanation=explanation_text)
            

        except Exception as e:
            self.result_label.config(text=f"Prediction error: {e}", fg="red")

    def on_reset(self):
        for feature, widget in self.inputs.items():
            if isinstance(widget, tk.Entry):
                widget.delete(0, tk.END)
            elif isinstance(widget, tk.StringVar):
                widget.set('')
        self.result_label.config(text="")


    def explain_prediction(self, pipeline, X_input):
        # Recupera modello e preprocessing
        model = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessing']
        feature_engineer = pipeline.named_steps.get('feature_transformer', None)
        
        X_feat = feature_engineer.transform(X_input)
        X_proc = preprocessor.transform(X_feat)

        if hasattr(X_proc, "toarray"):
                X_proc = X_proc.toarray()

        feature_names = preprocessor.get_feature_names_out()

        # Crea SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_proc)
        
        # Seleziona la classe predetta per questa istanza
        predicted_class = model.predict(X_proc)[0]
        shap_instance = shap_values[0]                     
        shap_feature_values = shap_instance[:, predicted_class]  

        # Crea dizionario delle feature con il valore SHAP
        shap_dict = dict(zip(feature_names, shap_feature_values))
        top_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        # Formatta testo
        explanation = "Top 5 influential features:\n"
        for name, val in top_features:
            val = float(np.ravel(val)[0]) if isinstance(val, np.ndarray) else float(val)
            direction = "↑" if val > 0 else "↓"
            explanation += f"{name}: {direction} impact ({val:.3f})\n"

        return explanation
    
    def show_prediction_details(self, model_name, probs, shap_explanation):

        win = tk.Toplevel(self.root)
        win.title(f"Prediction Details — {model_name}")
        win.geometry("450x400")

        # Probabilità
        prob_frame = tk.LabelFrame(win, text="Class Probabilities", font=("Helvetica", 11, "bold"))
        prob_frame.pack(padx=10, pady=10, fill="both")

        for label, prob in probs.items():
            tk.Label(prob_frame, text=f"{label}: {prob:.2%}", anchor="w", font=("Courier", 10)).pack(fill="x", padx=10)

        # Spiegazione SHAP
        shap_frame = tk.LabelFrame(win, text="Top SHAP Features", font=("Helvetica", 11, "bold"))
        shap_frame.pack(padx=10, pady=10, fill="both", expand=True)

        shap_text = tk.Text(shap_frame, wrap="word", font=("Courier", 10), height=10)
        shap_text.insert("1.0", shap_explanation)
        shap_text.configure(state="disabled")
        shap_text.pack(padx=10, pady=5, fill="both", expand=True)


   

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x900")
    app = StudentFormApp(root)
    root.mainloop()