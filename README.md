# Deep Learning Based Intrusion Detection with Power BI Analytics
## 🎥 Execution Video
[Watch Project Execution](https://drive.google.com/file/d/1-6ydSODV1STgsfLMX_HV_xAQQx8MrH6I/view?usp=sharing)

## Project Overview

This project implements a Deep Learning-based Intrusion Detection System for cloud environments and extends it with structured data processing and Power BI reporting.

The system detects malicious network activity using GAN and DBN models and exports structured prediction data for analytical dashboard visualization.

---

## Project Pipeline

Synthetic Network Log Data  
→ Data Preprocessing (Python, Pandas, NumPy)  
→ Model Training (DBN / GAN)  
→ Prediction Generation  
→ Structured CSV Export  
→ Power BI Dashboard Reporting  

---

## Models Implemented

- Deep Belief Network (DBN)
- Generative Adversarial Network (GAN)
- Logistic Regression Classifier

---

## Power BI Integration

The project generates a structured dataset:

powerbi_intrusion_dashboard_dataset.csv

This dataset contains:

- Actual Label  
- Predicted Label  
- Model Type (GAN / DBN)  
- Detection Status (Correct / Incorrect)  
- Confidence Score  
- Dataset Split  

This dataset is imported into Microsoft Power BI Desktop to create analytical dashboards for intrusion monitoring and model performance comparison.

---

## Dashboard Insights

- Model Accuracy Comparison (GAN vs DBN)
- Intrusion Detection Rate
- Detection Status Distribution
- Dataset Split Performance
- Confidence Score Analysis

---

## Technologies Used

Backend & Data Processing:
- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras

Data Visualization:
- Microsoft Power BI Desktop

---

## Project Structure

data/
    train_*.csv  
    test_*.csv  
    gan_detailed_results.csv  
    dbn_detailed_results.csv  
    powerbi_intrusion_dashboard_dataset.csv  

generate_datasets.py  
intrusion_dbn.py  
intrusion_gan.py  
export_powerbi_dataset.py  
compare_models.py  
requirements.txt  

---

## How to Run

1. Generate datasets:
   python generate_datasets.py

2. Train DBN model:
   python intrusion_dbn.py

3. Train GAN model:
   python intrusion_gan.py

4. Create Power BI dataset:
   python export_powerbi_dataset.py

5. Import the generated CSV file into Power BI Desktop to build dashboards.

---

## Key Highlights

- End-to-end intrusion detection pipeline
- Structured data transformation and export
- Model performance comparison
- Power BI dashboard integration
- Data engineering workflow for security analytics
