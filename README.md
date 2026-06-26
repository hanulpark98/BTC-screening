# BTC-vs-BTD Machine Learning Decision Support 🏥

This repository provides code and documentation for the study:

**A machine-learning decision-support model for differentiating biliary tract cancer from benign biliary tract disease at initial evaluation**

We developed and externally validated machine-learning models to support differentiation of **biliary tract cancer (BTC)** from **benign biliary tract disease (BTD)** using routinely available structured clinical and laboratory data collected at initial evaluation.

The model is intended to support BTC risk stratification and prioritization of further diagnostic evaluation in patients with suspected biliary tract disease. It is not intended to replace clinician judgment, imaging, endoscopy, pathology, or standard diagnostic workup.

---

## Overview

BTC and BTD can be difficult to distinguish during initial clinical evaluation because they may present with overlapping symptoms and laboratory abnormalities. This project evaluates whether structured clinical data available before downstream diagnostic confirmation can provide useful decision support.

The final models use **TabPFN** with two predictor sets:

- **Compact 20-feature model**: designed for portability across institutions
- **Extended 40-feature model**: includes additional specialist-oriented variables

---

## Main results

External validation was performed on an independent cohort from a separate institution.

| Model | AUROC | AUPRC | Brier score | Operating tendency |
|---|---:|---:|---:|---|
| Compact 20-feature TabPFN | 0.853 | 0.603 | 0.123 | More specificity-oriented |
| Extended 40-feature TabPFN | 0.848 | 0.620 | 0.174 | More sensitivity-oriented |

In the external subset with available CA19-9 measurements, both TabPFN models showed stronger discrimination than CA19-9 alone.

---

## Compact 20-feature model

The compact model uses variables expected to be routinely available at initial evaluation:

```text
Age, sex, height, weight,
diabetes, current smoking status, current alcohol use,
white blood cell count, hemoglobin, hematocrit, platelet count,
C-reactive protein, albumin, total bilirubin,
AST, ALT, alkaline phosphatase,
blood urea nitrogen, creatinine, glucose
```

Full feature definitions are provided in:

```text
docs/feature_definitions.md
docs/data_dictionary.md
```

---

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── environment.yml
├── data/
│   ├── README.md
│   └── synthetic_demo_data.csv
├── src/
│   ├── feature_sets.py
│   ├── preprocessing.py
│   ├── train_internal_validation.py
│   ├── evaluate_external.py
│   ├── metrics.py
│   └── shap_analysis.py
├── notebooks/
│   ├── 01_feature_set_overview.ipynb
│   ├── 02_internal_validation_demo.ipynb
│   └── 03_shap_interpretation_demo.ipynb
├── results/
│   └── figures/
└── docs/
    ├── intended_use.md
    ├── feature_definitions.md
    ├── data_dictionary.md
    ├── model_card.md
    ├── reproducibility.md
    └── limitations.md
```

---

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/btc-btd-initial-evaluation-ml.git
cd btc-btd-initial-evaluation-ml
```

Using conda:

```bash
conda env create -f environment.yml
conda activate btc-btd-ml
```

Or using pip:

```bash
pip install -r requirements.txt
```

---

## Quick start

Because the original clinical data are restricted, this repository includes synthetic example data for demonstrating the expected input format and code execution.

```bash
python src/evaluate_external.py \
    --input data/synthetic_demo_data.csv \
    --feature_set compact_20 \
    --output results/demo_predictions.csv
```

The synthetic data are provided only for demonstration and should not be used for clinical interpretation.

---

## Data availability

The original patient-level clinical data are not publicly available because they contain restricted hospital data.

Data may be available from the corresponding authors upon reasonable request, subject to institutional approval, data-use agreements, and relevant ethics regulations.

---

## Intended use

This repository is intended to support:

- Reproducibility of the modeling pipeline
- Retrospective validation
- External validation in independent cohorts
- Clinical machine-learning benchmarking
- Feature-set and model-comparison studies

The model should not be used as a standalone basis for diagnosis or treatment decisions. Prospective validation and appropriate clinical governance are required before routine clinical deployment.

---

## Citation

```bibtex
@article{park_btc_btd_ml_2026,
  title   = {A machine-learning decision-support model for differentiating biliary tract cancer from benign biliary tract disease at initial evaluation},
  author  = {Park, Hanul and Kim, Kibeom and Lee, Jonghyun and Han, Sung Yong and Cho, Chang Min and Lee, Dong Wook and Heo, Jun and Jung, Min Kyu and Nam, Hyeong Seok and Gahm, Jin Kyu and Song, Giltae and Kim, Dong Uk},
  journal = {To appear},
  year    = {2026}
}
```

Please update this citation after journal publication.

---

## Disclaimer

This repository is provided for academic and clinical AI research. The model has been retrospectively developed and externally validated, but it is not an approved medical device. It should not replace clinician judgment or established diagnostic workup.
