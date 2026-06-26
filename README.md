# ML-Based Risk assessment of Biliary Tract Cancer 🏥

This repository contains code and documentation for reproducing the machine-learning pipeline described in:

"A machine-learning decision-support model for differentiating biliary tract cancer from benign biliary tract disease at initial evaluation"

The model is intended for research use and retrospective validation only. It is not a standalone diagnostic device and should not be used to make clinical decisions without prospective validation and appropriate regulatory review.

## What this repository provides

- Definitions of the compact 20-feature and extended 40-feature predictor sets
- Model-development and validation code
- Metric computation scripts for AUROC, AUPRC, Brier score, sensitivity, specificity, PPV, and NPV
- SHAP-based model interpretation code
- Synthetic/example input files for demonstrating the pipeline
- Documentation of the intended use, limitations, and data-access restrictions

## What this repository does not provide

The original patient-level clinical dataset is not publicly released because it contains restricted clinical data from participating hospitals. Access may be considered upon reasonable request and institutional permission.

## 1. Required input clinical variables of the BTC-screening model ☑
The following variables are presented in order, each with its required unit value:

    1. Regular_daily_activity (hr/day)
    2. Fatty_Liver (present:1, absent:0)
    3. Vitamin (vitamin usage in past year; over a month: 1, otherwise: 0)
    4. Uric_Acid (mg/dL)
    5. HBsAb_negative (Yes: 1, No: 0)
    6. Hematocrit (%)
    7. HBsAb_positive (Yes: 1, No: 0)
    8. HBsAg_negative (Yes: 1, No: 0)
    9. Weight (kg)
    10. HCVAb_negative (Yes: 1, No: 0)
    11. Albumin (g/dL)
    12. AST (U/L)
    13. Height (cm)
    14. Platelets (10⁹/L)
    15. ALT (U/L)
    16. APTT (s)
    17. Alcohol_status (Current: 2, Former: 1, Never: 0)
    18. GGT (U/L)
    19. ESR (U/L)
    20. C Reactive Protein (mg/dL)
    21. Total Bilirubin (g/dL)
    22. Hemoglobin (g/dL)
    23. Age (years)
    24. Direct Bilirubin (mg/dL)
    25. ALP (U/L)
    26. CEA (ng/mL)

For any missing or unavailable input, the variable could be imputed with the median value from our training cohort (n = 891 benign biliary tract patients and n = 548 malignant biliary tract patients).

    1. Regular_daily_activity: 
    2. Fatty_Liver: 
    3. Vitamin: 
    4. Uric_Acid: 
    5. HBsAb_negative
    6. Hematocrit
    7. HBsAb_positive
    8. HBsAg_negative
    9. Weight
    10. HCVAb_negative
    11. Albumin
    12. AST
    13. Height
    14. Platelets
    15. ALT 
    16. APTT
    17. Alcohol_status
    18. GGT
    19. ESR 
    20. C Reactive Protein 
    21. Total Bilirubin
    22. Hemoglobin
    23. Age
    24. Direct Bilirubin
    25. ALP
    26. CEA 

## 2. Prepraring the model environment before running ⚓
2.1 Set up a Python development environment such as VS Code, PyCharm, or Jupyter Notebook

2.2 Inside the Python development environment, install all required dependencies by running:

    pip install -r requirements.txt

or manually installing all libraries

2.3 Download the file: 

    tabpfn_model.pkl 
    
to your workspace inside your Python development environment

## 3. Model usage 🙍👧
With the clinical variables prepared for a sample patient of suspection, run the file

    model_initiator.py 

