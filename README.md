# Sampling-techniques-for-imbalanced-datasets
# Sampling Techniques for Imbalanced Datasets

## Project Overview
This project analyzes the impact of different sampling techniques on the performance of various machine learning models when working with highly imbalanced datasets. A credit card fraud detection dataset is balanced using oversampling, followed by the application of five different sampling techniques and five machine learning models to evaluate classification accuracy.

## Objective
- To understand the importance of sampling techniques in handling imbalanced datasets.
- To compare the effect of different sampling strategies on model performance.
- To identify the best-performing model and sampling technique combination based on accuracy.

## Dataset
The project uses a highly imbalanced Credit Card Fraud Detection dataset.

Source: Creditcard_data.csv  
Class Distribution: The dataset contains a majority class (legitimate transactions) and a minority class (fraudulent transactions).

## Methodology

### 1. Data Preprocessing
- The imbalanced dataset was converted into a balanced dataset by oversampling the minority class.
- Five different samples were generated from the balanced dataset using distinct sampling techniques.

### 2. Sampling Techniques Used
- Sampling1: Simple Random Sampling
- Sampling2: Systematic Sampling
- Sampling3: Stratified Sampling
- Sampling4: Cluster Sampling (K-Means based)
- Sampling5: Bootstrap Sampling

### 3. Machine Learning Models
- M1: Logistic Regression
- M2: Decision Tree Classifier
- M3: Random Forest Classifier
- M4: K-Nearest Neighbors (KNN)
- M5: Naïve Bayes

## Implementation Details
- Each sampling technique was applied to generate a dataset of fixed size.
- Each model was trained and tested using a 75/25 train-test split.
- If a sampled dataset contained only one target class, training was skipped to avoid invalid model evaluation.

## Results
The following table shows the accuracy scores obtained for each model under different sampling techniques.

| Model | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|------|-----------|-----------|-----------|-----------|-----------|
| M1 | 0.7835 | 0.9381 | 0.8750 | 0.9691 | 0.9278 |
| M2 | 0.9588 | 0.9588 | 0.9792 | 0.9794 | 0.9897 |
| M3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| M4 | 0.9691 | 0.9072 | 0.9583 | 0.9897 | 0.9278 |
| M5 | 0.8247 | 0.6495 | 0.7292 | 1.0000 | 0.7423 |

## Key Observations
- Random Forest (M3) achieved perfect accuracy across all sampling techniques, indicating strong performance on the balanced dataset.
- Cluster Sampling (Sampling4) produced very high accuracy for multiple models, showing that it can be effective when clusters contain representative samples from both classes.
- Decision Tree (M2) performed consistently well across all sampling techniques.
- Naïve Bayes (M5) showed high variability in performance, indicating sensitivity to the sampling strategy.
- Tree-based models generally outperformed linear and probabilistic models.

## Discussion
- Random Forest emerged as the most robust model in this experiment, although perfect accuracy may indicate potential overfitting due to oversampling.
- Cluster Sampling proved highly effective in this run, producing strong results for multiple models.
- Naïve Bayes was the most sensitive model to sampling variations.
- Overall, combining tree-based models with Cluster or Bootstrap sampling yielded the best performance.

## Conclusion
- Best Overall Model: Random Forest (M3)
- Most Effective Sampling Technique in this run: Cluster Sampling (Sampling4)
- Most Stable Models: Decision Tree (M2) and Random Forest (M3)

## How to Run
1. Install required libraries:
   pip install pandas numpy scikit-learn
2. Download the dataset file.
3. Run the Python script or notebook.
4. The output will generate a CSV file named sampling_results.csv containing the accuracy results.

