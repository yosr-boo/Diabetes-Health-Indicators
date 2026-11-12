# Diabetes Risk Classification (CDC BRFSS 2015)

Six classification models compared on their ability to flag diabetes or prediabetes from self-reported survey answers.

## Question

How well can lifestyle and health indicators from a phone survey identify people with diabetes or prediabetes, and which model offers the best balance between catching cases (recall) and avoiding false alarms (precision)?

## Results

Test set: 30% of 70,692 respondents, classes balanced 50/50.

| Model | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| Gradient boosting | 0.755 | 0.796 | 0.734 | 0.764 |
| Neural network (2 hidden layers) | 0.748 | 0.847 | 0.704 | 0.769* |
| Logistic regression | 0.743 | 0.772 | 0.727 | 0.749 |
| K-nearest neighbours (k = 20) | 0.732 | 0.751 | 0.721 | 0.735 |
| Random forest (100 trees) | 0.724 | 0.755 | 0.709 | 0.731 |
| Decision tree | 0.663 | 0.645 | 0.665 | 0.655 |

\* Computed from the test-set precision and recall.

- Gradient boosting has the highest accuracy and precision.
- The neural network finds the most true cases (recall 0.85) at the price of more false positives.
- Logistic regression is within 1.2 points of the best model on accuracy while staying fully interpretable.
- In the logistic regression, high blood pressure, high cholesterol and poorer self-rated general health carry the largest positive coefficients.

## Data

[Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), derived from the CDC Behavioral Risk Factor Surveillance System 2015. This project uses the balanced binary file (`diabetes_binary_5050split_health_indicators_BRFSS2015.csv`): 70,692 respondents, 21 features, target `Diabetes_binary` (0 = no diabetes, 1 = prediabetes or diabetes).

The data files are not stored in this repository; download them from the link above into a `data/` folder.

## Method

1. Exploration: distributions, correlation heatmap, outlier check with the interquartile range.
2. Four features dropped (`AnyHealthcare`, `NoDocbcCost`, `Income`, `Education`), leaving 17.
3. 70/30 train/test split; min-max scaling fitted on the training set only.
4. K-means clustering (elbow and silhouette) as an unsupervised look at structure in the data.
5. Six supervised models, compared on accuracy, recall, precision and F1.
6. Feature importance from logistic regression coefficients and tree-based models.

## Limitations

- All inputs are self-reported survey answers, not clinical measurements.
- The balanced sample does not reflect real prevalence, so precision in a real population would be lower.
- Models use fixed hyperparameters, with a single train/test split and no cross-validation.
- This is a course project, not a screening tool.

## Tools

Python, pandas, NumPy, scikit-learn, Keras/TensorFlow, statsmodels, Matplotlib, Seaborn.

## Repository

```
├── diabetes_risk_models.ipynb   # full analysis
├── requirements.txt
└── README.md
```
