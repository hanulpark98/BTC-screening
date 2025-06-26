import pickle
import pandas as pd
import numpy as np
import shap
from sklearn.metrics import brier_score_loss

FEATURES = [
    'Regular_daily_activity(1hr/day)', 'Fatty_Liver', 'Vitamin', 'Uric_Acid',
    'HBsAb_negative', 'Hematocrit', 'HBsAb_positive', 'HBsAg_negative', 'weight',
    'HCVAb_negative', 'Albumin', 'AST', 'height', 'Platelets', 'ALT',
    'APTT', 'Alcohol_status', 'GGT', 'ESR',
    'C Reactive Protein', 'Total Bilirubin', 'Hemoglobin', 'age',
    'Direct Bilirubin', 'ALP', 'CEA'
]


def load_model(path: str):
    """Load the trained TabPFN model from a pickle or joblib file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def prompt_features():
    """Prompt the user to input each clinical feature value."""
    values = {}
    print("Please enter the following patient data:")
    for feat in FEATURES:
        while True:
            val = input(f"  - {feat}: ")
            try:
                # try numeric conversion
                if '.' in val:
                    values[feat] = float(val)
                else:
                    values[feat] = int(val)
                break
            except ValueError:
                print("    Invalid number. Please enter again.")
    return pd.DataFrame([values])


def main():
    # Load model
    model_path = input("Enter path to your model file (.pkl): ")
    model = load_model(model_path)

    # Collect patient features
    df = prompt_features()

    # Predict risk probability for positive class
    prob = model.predict_proba(df)[:, 1][0]
    print(f"\nEstimated risk of BTC: {prob:.4f}")

    # # Optional: ask for true label to compute sample Brier score
    # label_known = input("Do you know the true diagnosis? (y/n): ")
    # if label_known.lower().startswith('y'):
    #     y = None
    #     while y not in ('0', '1'):
    #         y = input("  Enter true label (0=benign, 1=malignant): ")
    #     y = int(y)
    #     brier = (y - prob) ** 2
    #     print(f"Sample Brier score (lower→more reliable): {brier:.4f}")

    # # Optional: SHAP explanation
    # explain = input("Show SHAP feature contributions? (y/n): ")
    # if explain.lower().startswith('y'):
    #     explainer = shap.KernelExplainer(model.predict_proba, df)
    #     shap_vals = explainer.shap_values(df)
    #     print("\nSHAP values for positive class:")
    #     for feat, sv in zip(FEATURES, shap_vals[1][0]):
    #         print(f"  {feat}: {sv:.4f}")

if __name__ == '__main__':
    main()
