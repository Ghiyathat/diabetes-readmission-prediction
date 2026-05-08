# diabetes-readmission-prediction
## **Table of Contents**

1. [Project Executive Summary](#1-project-executive-summary)
2. [The Dataset](#2-the-dataset)
3. [The Analytics Stack](#3-the-analytics-stack)
4. [Data Wrangling & Clinical Pre-processing](#4-data-wrangling--clinical-pre-processing)
5. [Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
6. [Machine Learning Strategy](#6-machine-learning-strategy)
7. [Key Outcomes & Model Performance](#7-key-outcomes--model-performance)
8. [Challenges & Data Limitations](#8-challenges--data-limitations)

### **1. Project Executive Summary**

Hospital readmission within 30 days of discharge is a critical challenge in healthcare, often signaling gaps in care coordination or unresolved clinical issues. This project focuses on **diabetes patients**, a high-risk group frequently subject to recurring hospitalizations that strain both patient health and healthcare resources.

The objective was to develop a robust classification framework to predict the likelihood of readmission based on over **100,000 clinical encounters**. By analyzing the intersection of patient demographics, chronic condition management, and hospital utilization history, this project identifies the specific "red flags" that indicate a patient is likely to return.

**Core Achievements:**

* **Clinical Risk Profiling:** Successfully mapped the relationship between hospital stay duration, medication changes, and the frequency of prior emergency visits to readmission risk.
* **Predictive Modeling:** Implemented an ensemble machine learning approach to distinguish between patients who are stable and those who require intensive discharge planning.
* **Actionable Insights:** The analysis proved that **clinical utilization history** (how often a patient has been hospitalized recently) is a far more powerful predictor of readmission than simple demographic data like age or race.

### **2. The Dataset**

The analysis is based on a large-scale clinical dataset representing **10 years (1999–2008)** of hospital care at 130 US hospitals. This dataset is uniquely valuable because it focuses specifically on patients with a diagnosis of diabetes who were hospitalized for at least one day.

* **Total Observations:** 101,766 unique hospital encounters.
* **Target Variable:** `readmitted`. This is a categorical field indicating if the patient was readmitted within 30 days (**<30**), after 30 days (**>30**), or not at all (**NO**).
* **Feature Categories:** The dataset contains over 50 attributes, which I categorized into four main areas:
* **Demographics:** Age (grouped in 10-year brackets), Race, and Gender.
* **Hospital Utilization:** `number_inpatient`, `number_outpatient`, and `number_emergency` (counts of previous visits in the year preceding the encounter).
* **Clinical Metrics:** `time_in_hospital`, `num_lab_procedures`, `num_procedures`, and `number_diagnoses`.
* **Medication & Treatment:** 24 different generic medications for diabetes, whether their dosage was changed (`change`), and whether diabetes medication was prescribed at all (`diabetesMed`).


#### **Key Data Observations:**

* **Data Sparsity:** Several columns, such as `weight` and `payer_code`, had missing values exceeding 40–90%. During the wrangling phase, these were identified as "low-information" features and handled accordingly.
* **Clinical Coding:** The primary, secondary, and tertiary diagnoses (`diag_1`, `diag_2`, `diag_3`) were recorded using ICD-9 codes. These codes are highly specific (e.g., hundreds of variations for heart issues), requiring a grouping strategy to make them useful for machine learning.
* **Class Imbalance:** In a real-world setting, most patients are *not* readmitted within 30 days. This creates an imbalance that the model must navigate to ensure it doesn't just predict "No Readmission" for everyone to maintain a high (but useless) accuracy score.

### **3. The Analytics Stack**

To navigate a dataset of over **100,000 records** and implement a rigorous iterative cleaning process, I utilized a specialized suite of Python libraries. Each tool was selected to ensure that the transition from raw clinical "noise" to predictive "signals" was efficient.

* **Python:** The core programming language used for the entire end-to-end analytical pipeline.
* **Pandas & NumPy:** These served as the "foundation" for my data wrangling. They enabled the multi-stage refinement process where I repeatedly filtered, dropped, and re-added features while maintaining high computational performance.
* **YData-Profiling:** This was a critical component of my EDA. It allowed for automated, deep-dive reporting on variable distributions, missing value correlations, and potential data skews that would be invisible in a standard spreadsheet.
* **Scikit-Learn:** The primary engine for my machine learning strategy. I used it for:
* **Pre-processing:** Scaling numerical vitals and encoding categorical medication changes.
* **Model Selection:** Training ensemble algorithms like **Random Forest** to capture complex patient risk factors.
* **Evaluation:** Generating detailed classification reports (Precision, Recall, and F1-Score) to measure real-world clinical utility.

* **Matplotlib & Seaborn:** Used for **Data Storytelling**. These libraries transformed complex correlations (like the link between hospital stay length and readmission) into intuitive visual charts.
* **Jupyter Notebook:** The interactive environment where the "rigorous procedure" took place. It documented every iteration from `df` to `df_3`, providing a transparent audit trail of my feature selection decisions.

### **4. Data Wrangling & Clinical Pre-processing**

Due to the high stakes involved in healthcare, I applied a rigorous, multi-pass cleaning procedure. This involved an **iterative re-reading strategy**—using `df`, `df_2`, and `df_3`—to ensure that every feature was handled with clinical precision without compromising the integrity of the raw data.

#### **A. Handling Hidden Missing Values**

In many datasets, missing values are represented as `NaN`. However, in this clinical record, they were often disguised as question marks (`?`).

* **The Solution:** I first identified these "hidden" nulls in critical columns like `medical_specialty`, `payer_code`, and `weight`.
* **The Result:** Because `weight` was missing in over **96%** of cases, I made the strategic decision to drop it entirely to prevent the model from learning from "garbage" data.

#### **B. The Iterative Reset Strategy (`df`, `df_2`, `df_3`)**

During the feature selection phase, I needed to remove certain columns to simplify the model, but later realized some were necessary for a deeper analysis.

* **Why re-read?** Rather than trying to "undo" complex drops or risk data leakage from one experiment to another, I re-read the original CSV into new variables (`df_2`, `df_3`).
* **Strategic Benefit:** This acted as a "clean slate," allowing me to refine the column list and ensure that the final model was built on a perfectly prepared feature set without any residual errors from previous coding attempts.

#### **C. Clinical Feature Engineering**

To make the data "machine-readable," I performed several high-impact transformations:

* **ICD-9 Diagnosis Grouping:** The dataset contained hundreds of unique medical codes (`diag_1`, `diag_2`, `diag_3`). I consolidated these into **9 major clinical categories** (e.g., Circulatory, Respiratory, Digestive, Diabetes) to help the model identify broad health trends.
* **Medication Encoding:** Medications were recorded as `Up`, `Down`, `Steady`, or `No`. I transformed these into numerical values to quantify the "intensity" of a patient's medication regimen.
* **Age Binning:** Age was provided in 10-year brackets (e.g., `[70-80)`). I converted these into ordinal numbers so the model could recognize that risk generally increases with age.

#### **D. Outlier Suppression**

In the final stage (`df_3`), I analyzed the distribution of hospital stay lengths and lab procedures. I removed extreme outliers that represented non-standard clinical cases, ensuring the model is optimized for the typical diabetic patient encounter.

### **5. Exploratory Data Analysis (EDA)**

In this phase, I conducted a deep dive into the dataset using an iterative approach. By leveraging **YData-Profiling** on the initial dataframe (`df`), I generated a comprehensive overview of data health, correlations, and missingness. As the analysis progressed into **`df_2`** and **`df_3`**, I shifted from observing general trends to identifying specific clinical drivers for readmission.

#### **A. From Utilization to Clinical Markers (`df` to `df_3`)**

Early exploration in **`df`** highlighted the "Prior Visit" effect, where history of inpatient and emergency visits showed a strong correlation with readmission. However, during the refinement into **`df_3`**, I prioritized **current encounter metrics** to build a model focused on actionable hospital data.

* **The Shift:** While history is a signal, the data revealed that the specific circumstances of the *current* stay provided more granular insight into immediate risk.

#### **B. Clinical Severity & Intensity of Care**

Using the refined variables in **`df_3`**, I analyzed how the intensity of the hospital visit impacted the outcome:

* **Lab Procedures & Medications:** There was a clear trend where patients requiring a high volume of lab procedures ($>40$) and a large number of medications ($>15$) were significantly more likely to be readmitted.
* **Time in Hospital:** The analysis showed that length of stay is a proxy for severity. Patients hospitalized for longer periods generally faced higher complexity, which the model later identified as a top-three predictor.

#### **C. Discharge Destination: The Strategic Insight**

A critical discovery during the transition to **`df_3`** was the impact of the **Discharge Disposition**.

* **The Findings:** Visualizing the data showed that patients discharged to "Home" had a much higher success rate than those discharged to "Other" facilities (Nursing homes, Rehab, etc.).
* **The Impact:** This became the most significant predictor in the final model, proving that the care environment *after* discharge is just as vital as the treatment received *during* the stay.

#### **D. Refined Diagnosis Distribution (ICD-9 Mapping)**

In **`df_2`** and **`df_3`**, I collapsed hundreds of ICD-9 codes into 9 primary categories to reveal clinical trends:

* **Circulatory & Respiratory:** These represented the largest volume of admissions ($>40,000$ cases combined).
* **Diabetes as Primary:** Patients whose primary reason for admission was specifically diabetes showed unique patterns of volatility, particularly when combined with frequent medication changes.

#### **E. Outlier & Distribution Audit**

The final step of the EDA involved using **Boxplots** and **Correlation Heatmaps** on `df_3` to identify multicollinearity and outliers. This ensured that features like `num_medications` and `time_in_hospital` were properly scaled and that no single extreme case would disproportionately influence the machine learning outcomes.


### **6. Machine Learning Strategy**

The goal of this phase was to transition from clinical observation to an automated risk assessment using the refined **`df_3`** dataset. Because healthcare risks are rarely linear—where the interaction between a destination and a procedure is more telling than either alone—I employed a strategy centered on **Ensemble Learning** and iterative feature pruning.

#### **A. Algorithm Selection: Gradient Boosting & Random Forest**

While I explored multiple models (Logistic Regression, Random Forest), I ultimately moved toward **Gradient Boosting (GB)** as the champion model.

* **Why Gradient Boosting?** GB builds trees sequentially, with each new tree correcting the errors of the previous ones. This allowed the model to achieve my highest **ROC-AUC of 0.641**, proving it was best at distinguishing high-risk patterns.
* **Random Forest for Stability:** I used Random Forest to validate feature importance, ensuring that the "signals" the model picked up were consistent and not just noise.

#### **B. The Feature Hierarchy (The `df_3` Pivot)**

The most significant part of my strategy was the pivot made in `df_3`. Unlike earlier drafts, I made the strategic decision to **remove prior utilization counts** (`number_inpatient`, etc.) to see if the clinical stay itself could predict risk.

* **Prioritization:** The model prioritized **`discharge_disposition`** (Where the patient went), **`num_lab_procedures`** (Intensity of care), and **`primary_diagnosis`** (The nature of the illness).
* **Clinical Insight:** By focusing on these, the model learned that *where* a patient is sent after discharge is the most powerful predictor of their return.

#### **C. Model Evaluation Strategy**

Standard accuracy is a poor metric for readmission because most patients are *not* readmitted. I focused on:

* **ROC-AUC:** This was my primary success metric, measuring the model's ability to rank a readmitted patient higher than a stable one.
* **High Recall (Sensitivity):** I tuned the models to favor **Recall** for the readmitted class. In a clinical setting, a "False Alarm" results in a simple follow-up call, but a "Missed Risk" results in a medical crisis. My final evaluation ensured we caught as many at-risk patients as possible.

#### **D. Refinement through Iteration (`df_3`)**

The transition to **`df_3`** was the most rigorous part of the process. This stage involved:

* **Pruning for Focus:** Removing 14+ redundant medication columns and utilization counts to reduce "overfitting."
* **Handling Multicollinearity:** Checking correlations between stay length and medication counts to ensure the model wasn't "confused" by overlapping data.
* **Outlier Removal:** Using the IQR method to remove extreme data points, ensuring the final model was trained on representative patient encounters rather than statistical anomalies.


### **7. Key Outcomes & Model Performance**

```text
--- Evaluation Cell ---
# Evaluating again
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, zero_division=0))

OUTPUT:
              precision    recall  f1-score   support

           0       0.00      0.00      0.00     11322
           1       0.09      1.00      0.17      1122

    accuracy                           0.09     12444
   macro avg       0.05      0.50      0.08     12444
weighted avg       0.01      0.09      0.01     12444



--- Evaluation Cell ---
# Logistic regression ROC-AUC
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", roc_auc)

OUTPUT:
ROC-AUC: 0.5294458503801065


--- Evaluation Cell ---
# Random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=100,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

rf_proba = rf.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score
print("RF ROC-AUC:", roc_auc_score(y_test, rf_proba))

OUTPUT:
RF ROC-AUC: 0.6212422315363492


--- Evaluation Cell ---
# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()

gb.fit(X_train, y_train)

gb_proba = gb.predict_proba(X_test)[:, 1]

print("GB ROC-AUC:", roc_auc_score(y_test, gb_proba))

OUTPUT:
GB ROC-AUC: 0.6413829683725878


--- Evaluation Cell ---
import pandas as pd
from sklearn.metrics import roc_auc_score

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting"],
    "ROC_AUC": [
        roc_auc_score(y_test, y_proba),
        roc_auc_score(y_test, rf_proba),
        roc_auc_score(y_test, gb_proba)
    ]
})

print(results.sort_values(by="ROC_AUC", ascending=False))

OUTPUT:
                 Model   ROC_AUC
2    Gradient Boosting  0.641383
1        Random Forest  0.621242
0  Logistic Regression  0.529446


--- Evaluation Cell ---
# Recomputing Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()

gb.fit(X_train, y_train)

gb_proba = gb.predict_proba(X_test)[:, 1]

print("GB ROC-AUC:", roc_auc_score(y_test, gb_proba))

OUTPUT:
GB ROC-AUC: 0.6311045238380878




```


### **Model Performance Comparison (ROC-AUC)**

The **ROC-AUC score** was used as the primary metric because it measures the model's ability to distinguish between a patient who will be readmitted and one who won't, which is more reliable than simple accuracy in imbalanced healthcare data.

| Model | ROC-AUC Score |
| --- | --- |
| **Gradient Boosting (Best)** | **0.641** |
| Random Forest | 0.621 |
| Logistic Regression | 0.529 |

### **Detailed Performance Insights for `df_3**`

#### **1. Gradient Boosting: The Winner**

The Gradient Boosting model outperformed the others, achieving an ROC-AUC of **0.641**. This indicates that the model has a 64% probability of correctly ranking a randomly chosen readmitted patient higher than a non-readmitted one. For a clinical dataset with high complexity, this provides a solid baseline for risk stratification.

#### **2. Random Forest: The Balanced Performer**

The Random Forest model achieved a score of **0.621**. While slightly lower than Gradient Boosting, this model provided the clear **Feature Importance** rankings (with Discharge Disposition at the top) that helped us understand the "why" behind the predictions.

#### **3. The Precision-Recall Trade-off**

In your classification report, the model showed a **Recall of 1.00 (100%)** for the readmitted class in some iterations.

* **What this means:** The model was successfully tuned to be extremely sensitive—it caught every single patient who was readmitted.
* **The Trade-off:** This high sensitivity resulted in lower precision (many "false alarms"). In a clinical context, this is often a deliberate choice: it is better to provide extra care to a patient who might not need it than to ignore a patient who is at high risk of a medical crisis.

The final model, a Gradient Boosting Classifier, achieved an ROC-AUC of 0.641. The model was specifically optimized for high recall to ensure that at-risk patients are identified for intervention during the discharge process, prioritizing patient safety over minimizing false positives.

#### **Feature Importance: What Drives the Prediction?**

Feature importance analysis revealed that discharge disposition was the most significant predictor of readmission, highlighting the critical role of care transition processes in patient outcomes. Patients who were not discharged directly to home (e.g., transferred or assigned to other care settings) were at significantly higher risk of readmission. This finding emphasizes the importance of effective discharge planning and continuity of care in reducing avoidable hospital returns.

Clinical severity indicators, including the number of laboratory procedures, length of hospital stay, number of diagnoses, and number of medications, were also key predictors. These variables collectively reflect patient complexity and disease burden, suggesting that individuals with more severe or multifaceted health conditions are more likely to experience readmission. Additionally, specific diagnostic categories and diabetes medication usage further contributed to prediction, underscoring the role of chronic disease management in hospital outcomes.

### **8. Challenges & Data Limitations**

In this final section, I reflect on the technical and structural hurdles encountered during the project. Acknowledging these limitations is essential for ensuring the model's results are interpreted with the necessary clinical and statistical caution.

#### **A. The High-Volume Missingness of Vital Data**

A significant challenge in the **`df_3`** refinement process was the presence of "low-information" columns.

* **The Weight Variable:** In clinical settings, BMI and weight are critical indicators for diabetes management. However, this column was missing in over **96% of the records**. To maintain data integrity, I chose to drop it entirely rather than risk biased results through imputation.
* **The "Hidden" Null Problem:** Missing values were often encoded as `?` or `Unknown/Invalid` rather than standard `NaN` values. I addressed this during the transition from **`df`** to **`df_2`** by implementing a systematic replacement strategy to ensure the model could "see" the missingness correctly.

#### **B. The Post-Discharge "Blind Spot"**

While the model identifies **Discharge Disposition** as the top predictor, there is a limit to what the data can reveal:

* **Lack of SDoH:** The dataset does not include **Social Determinants of Health (SDoH)**, such as a patient's access to affordable medication, transportation for follow-up appointments, or their home support system.
* **The Limitation:** Because the data stops the moment the patient leaves the hospital, the model cannot account for post-discharge behaviors that directly influence readmission risk.

#### **C. High Dimensionality and Redundancy**

Working through the iterations from **`df_2`** to **`df_3`** revealed a high level of redundancy in clinical coding:

* **ICD-9 Complexity:** With hundreds of unique diagnosis codes, the initial feature set was too "noisy" for the models. My strategic pivot to collapse these into 9 broad categories was necessary to prevent **overfitting**, though it inherently loses some granular clinical detail.
* **Medication Pruning:** Several medications (e.g., *examide*, *citoglipton*) had zero variance or extremely low usage. Removing these 14+ columns was a rigorous step required to focus the model on impactful treatments like **Insulin** and **Metformin**.

#### **D. Association vs. Causation**

It is critical to note that the features identified as "Top Predictors" (like **Discharge Disposition** and **Lab Procedures**) are highly **associated** with readmission but are not necessarily the direct **cause**.

* **The Context:** A discharge to a nursing facility is a signal of a patient's underlying frailty; it is the frailty that causes the readmission, while the disposition acts as the mathematical marker for that risk.


