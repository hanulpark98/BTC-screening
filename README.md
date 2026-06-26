# BTC-vs-BTD Machine Learning Decision Support 🏥

This repository provides code and documentation for the study:

**A machine-learning decision-support model for differentiating biliary tract cancer from benign biliary tract disease at initial evaluation**

We developed and externally validated machine-learning models to support differentiation of biliary tract cancer (BTC) from benign biliary tract disease (BTD) using routinely available structured clinical and laboratory data collected at initial evaluation.

The model is intended to support BTC risk stratification and prioritization of further diagnostic evaluation in patients with suspected biliary tract disease. It is not intended to replace clinician judgment, imaging, endoscopy, pathology, or standard diagnostic workup.


## Overview ☑

BTC and BTD can be difficult to distinguish during initial clinical evaluation because they may present with overlapping symptoms and laboratory abnormalities. This project evaluates whether structured clinical data available before downstream diagnostic confirmation can provide useful decision support.

The final models use TabPFN with two predictor sets:

Compact 20-feature model: designed for portability across institutions
Extended 40-feature model: includes additional specialist-oriented variables














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

