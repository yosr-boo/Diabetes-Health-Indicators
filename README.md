# Diabetes Risk Classification (CDC BRFSS 2015)

Six classification models compared on their ability to flag diabetes or prediabetes from self-reported survey answers.

## Question

How well can lifestyle and health indicators from a phone survey identify people with diabetes or prediabetes, and which model offers the best balance between catching cases (recall) and avoiding false alarms (precision)?

## Results

Test set: 30% of 70,692 respondents, classes balanced 50/50.

| Model | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| Gradient boosting | 0.755 | 0.796 | 0.734 | 0.764 |
| Neural network (2 hidden layers) | 0.753 | 0.794 | 0.732 | 0.761 |
| Logistic regression | 0.749 | 0.768 | 0.737 | 0.752 |
| K-nearest neighbours (k = 20) | 0.732 | 0.758 | 0.718 | 0.738 |
| Random forest (100 trees) | 0.724 | 0.756 | 0.708 | 0.731 |
| Decision tree | 0.662 | 0.645 | 0.665 | 0.655 |

- Gradient boosting performs best overall, with the highest accuracy, recall and F1 score.
- The neural network is a close second on every metric.
- Logistic regression has the highest precision and is within one point of the best accuracy while staying fully interpretable.
- In the logistic regression (features scaled to 0–1, so coefficients are comparable), BMI, poorer self-rated general health and age carry the largest positive coefficients, followed by cholesterol check, high blood pressure and high cholesterol.

## Data

[Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), derived from the CDC Behavioral Risk Factor Surveillance System 2015. This project uses the balanced binary file (`diabetes_binary_5050split_health_indicators_BRFSS2015.csv`): 70,692 respondents, 21 features, target `Diabetes_binary` (0 = no diabetes, 1 = prediabetes or diabetes).

The data files are not stored in this repository; download them from the link above into a `data/` folder.

## Method

1. Exploration: distributions, correlation heatmap, outlier check with the interquartile range.
2. Four features dropped (`AnyHealthcare`, `NoDocbcCost`, `Income`, `Education`), leaving 17.
3. 70/30 train/test split; min-max scaling fitted on the training set only.
4. K-means clustering (elbow and silhouette) on the scaled training features, as an unsupervised look at structure in the data.
5. Six supervised models, all evaluated the same way on the test set: accuracy, recall, precision and F1 at a 0.5 threshold.
6. Feature importance from logistic regression coefficients and tree-based models.

## Limitations

- All inputs are self-reported survey answers, not clinical measurements.
- The balanced sample does not reflect real prevalence, so precision in a real population would be lower.
- Models use fixed hyperparameters, with a single train/test split and no cross-validation.
- This is a course project, not a screening tool.

## Tools

Python, pandas, NumPy, scikit-learn, Keras, Matplotlib, Seaborn.

## Repository

```
├── diabetes_risk_models.ipynb   # full analysis
├── requirements.txt
└── README.md
```
