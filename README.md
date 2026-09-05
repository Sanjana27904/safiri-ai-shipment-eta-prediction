# Predicting Shipment ETA and Delay Propagation Using AI

A machine learning project for predicting shipment arrival delays and understanding how delays during a logistics journey can affect the final ETA.

This project was developed as part of the **Safiri AI Take-Home Assignment**.

## Project Overview

In logistics, a shipment can be delayed due to factors such as departure delays, ports, routes, and changes in the estimated arrival time. A delay at an earlier stage does not always directly translate into the same delay at the final destination.

The goal of this project is to:

* Predict the final arrival delay of a shipment.
* Identify the factors that are most useful for predicting delays.
* Analyze how earlier delays relate to the final arrival delay.
* Provide a simple way to make predictions for a new shipment.

## Dataset

The project uses a maritime vessel tracking dataset containing information about vessel movements, ports, scheduled times, estimated times, and actual times.

The original dataset was large, so a sample of **250 records** was prepared for this project.

### Main Features

* Departure and arrival ports
* Route
* Ship
* Scheduled departure and arrival times
* Estimated arrival time
* Actual departure and arrival times
* Departure delay
* Scheduled transit duration
* Scheduled departure hour and day of week

### Target

The target variable is:

```text
Arrival Delay Hours = Actual Arrival − Scheduled Arrival
```

A positive value represents a late arrival, while a negative value represents an arrival slightly earlier than scheduled.

## Feature Engineering

A few additional features were created from the available timestamps:

```text
Departure Delay = Actual Departure − Scheduled Departure

Scheduled Transit Hours = Scheduled Arrival − Scheduled Departure

ETA Update Gap = Estimated Arrival − Scheduled Arrival
```

The **ETA Update Gap** helps capture how much the latest estimated arrival differs from the original schedule.

To avoid data leakage, information that would only be available after the shipment had completed its journey, such as actual arrival and actual transit duration, was not used as model input.

## Approach

I used a **Random Forest Regressor** for the prediction task.

Random Forest was chosen because it can capture non-linear relationships between different shipment features without requiring a simple linear relationship.

The categorical features such as ship, ports, and route were encoded before training.

### Training Setup

* Train/Test Split: **80/20**
* Number of Trees: **300**
* Maximum Depth: **8**
* Minimum Samples per Leaf: **2**
* Random State: **42**

I also created a simple baseline where the final arrival delay is assumed to be approximately equal to the departure delay. This was used to compare the machine learning model against a simpler approach.

## Results

The Random Forest model achieved the following results on the test set:

| Metric |    Random Forest |     Baseline |
| ------ | ---------------: | -----------: |
| MAE    | **0.0733 hours** | 0.4391 hours |
| RMSE   | **0.1019 hours** | 0.5082 hours |
| R²     |       **0.8124** |            — |

The model's MAE of **0.0733 hours is approximately 4.4 minutes**.

The baseline MAE was approximately **26.3 minutes**, so the Random Forest model performed considerably better on the test data.

## Delay Factors and Explainability

I examined the feature importance from the Random Forest model to understand which factors were most useful for the prediction.

The main feature groups were:

| Feature Group  | Importance |
| -------------- | ---------: |
| ETA Update     | **27.64%** |
| Arrival Port   | **19.51%** |
| Departure Port | **19.12%** |
| Route          | **17.87%** |
| Scheduled Time |  **4.80%** |
| Ship           |  **1.03%** |

The ETA update gap was the strongest individual signal. Port and route information were also important, suggesting that different routes and ports can have different operational patterns.

Feature importance indicates which features were useful to the model, but it does not prove that a feature directly causes the delay.

## Delay Propagation Analysis

To understand delay propagation, I compared departure delay with the final arrival delay.

The correlation between:

```text
Departure Delay → Arrival Delay = 0.118
```

This indicates a weak positive relationship.

In comparison:

```text
ETA Update Gap → Arrival Delay = 0.466
```

This shows that updated ETA information provides a stronger signal about the final delay in this dataset.

This suggests that final arrival delay should not simply be treated as a copy of departure delay. Other factors, including ETA updates, ports, and routes, also contribute to the final outcome.

## Example Prediction

The `predict.py` script can be used to make a prediction for a new shipment.

Example output:

```text
Predicted delay: 0.66 hours
Predicted ETA: 2026-09-08 16:39
```

The predicted delay is approximately **40 minutes**.

## Project Structure

```text
safiri-ai-shipment-eta-prediction/
│
├── data/
│   ├── safiri_final_shipment_eta_dataset_250.csv
│   ├── predictions.csv
│   ├── feature_importance.csv
│   └── eta_prediction_model.pkl
│
├── src/
│   ├── train_model.py
│   └── predict.py
│
├── notebook/
│   └── eta_analysis.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Sanjana27904/safiri-ai-shipment-eta-prediction.git
cd safiri-ai-shipment-eta-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you use `uv`, you can also install the dependencies with:

```bash
uv pip install -r requirements.txt
```

### 4. Train the model

From the `src` directory:

```bash
cd src
python train_model.py
```

This trains the Random Forest model and generates the prediction and feature importance files.

### 5. Make a prediction

From the same `src` directory:

```bash
python predict.py
```

This loads the saved model and predicts the arrival delay and ETA for the example shipment.

## Notebook

The Jupyter notebook contains the analysis performed during the project, including:

* Dataset exploration
* Delay correlations
* Feature importance
* Route and port analysis
* Delay propagation analysis
* Example prediction
* Final model results

To open the notebook:

```bash
jupyter notebook
```

Then open:

```text
notebook/eta_analysis.ipynb
```

## Limitations

The dataset does not contain some real-world factors that can affect shipment delays, such as:

* Weather and sea conditions
* Port congestion
* Customs clearance
* Cargo handling
* Last-mile transportation

The project also uses a sample of 250 records. A larger dataset covering more vessels, routes, ports, and time periods would provide more variation and could improve generalization.

## Future Improvements

With more data, the system could be improved by adding real-time vessel information, weather conditions, port congestion, vessel-specific historical patterns, and more detailed stage-level timestamps.

I would also consider testing gradient boosting or time-series models and using techniques such as **SHAP** for more detailed explanations of individual predictions.

## Conclusion

This project demonstrates a machine learning approach for predicting shipment arrival delays using operational, timing, port, and route-related information.

The Random Forest model achieved an **MAE of 0.0733 hours and an R² of 0.8124**, performing considerably better than the simple departure-delay baseline.

The analysis also showed that ETA updates, ports, routes, and departure delays provide useful signals for understanding final shipment delays and how they propagate through a logistics journey.
