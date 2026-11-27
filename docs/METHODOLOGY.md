# AeroAI Methodology

## Overview

AeroAI is an AI-enabled air quality intelligence platform designed to demonstrate an automated workflow for air-quality data collection, AQI processing, prediction, monitoring, anomaly detection, and intelligent retraining.

## Methodology Pipeline

The project follows the following high-level workflow:

1. Air-quality data collection
2. Data freshness and quality validation
3. Daily and hourly preprocessing
4. CPCB-method AQI calculation
5. Next-day AQI prediction
6. Data and model monitoring
7. Hourly anomaly detection
8. Pollution episode aggregation
9. Intelligent retraining decision
10. Candidate model validation before model replacement

## Data Sources

AeroAI uses multiple data sources for different purposes.

### CPCB Station Data

CPCB-linked station observations are used for current air-quality information when the source is available.

### OpenAQ

OpenAQ API v3 provides current and historical sensor measurements for the Doon University monitoring location in Dehradun.

OpenAQ measurements are also processed using CPCB AQI calculation rules to produce a CPCB-method AQI estimate.

These calculated values are not presented as official CPCB AQI.

### Historical API Data

The original daily and hourly historical data source is retained for the existing prediction and monitoring pipeline.

A freshness check prevents old observations from being treated as current data.

## Safety and Validation

AeroAI does not automatically continue processing when required input data is stale.

Dependent stages are skipped when their required source data is not fresh.

The OpenAQ monitoring branch remains separate from the existing model-training pipeline.

Experimental candidate models are not automatically promoted to the active model.

## Current Development Status

The project includes automated tests for data collection, preprocessing, AQI calculation, prediction safeguards, monitoring, pipeline orchestration, and stale-data handling.

Further methodology documentation will describe each stage of the pipeline in detail.

---

# Pipeline Design

## Pipeline Design Overview

AeroAI uses a multi-branch automated pipeline rather than a single linear workflow. Different data sources are processed independently according to their purpose and freshness.

The pipeline separates:

- current CPCB-linked AQI retrieval,
- OpenAQ live sensor monitoring,
- the original daily prediction workflow,
- the original hourly monitoring workflow, and
- the independent OpenAQ historical monitoring workflow.

This separation allows one branch to remain operational even when another data source is unavailable or stale.

## Freshness and Dependency Control

Before dependent processing stages are executed, AeroAI checks whether the required source data is sufficiently recent.

If an input is stale, only the stages that depend on that input are skipped. Independent branches are allowed to continue when their own required inputs remain healthy.

For example, stale hourly historical data can block hourly preprocessing and anomaly analysis without unnecessarily blocking an otherwise healthy daily prediction workflow.

## Pipeline Status Handling

AeroAI uses explicit execution statuses to distinguish data-source problems from software failures.

| Status | Meaning |
| --- | --- |
| SUCCESS | The stage completed successfully and its required output was generated or validated. |
| STALE_DATA | Data was available, but the newest observation was older than the permitted freshness limit. |
| SOURCE_UNAVAILABLE | The required external source or API could not provide usable data. |
| SKIPPED_STALE_INPUT | The stage was intentionally skipped because a required upstream input was stale or unhealthy. |
| SKIPPED_UPSTREAM_FAILURE | The stage was intentionally skipped because a required upstream stage failed. |
| FAILED | The stage encountered an execution or processing error. |
| PARTIAL_SUCCESS | Some independent branches completed successfully while other branches were stale, unavailable, or skipped. |

The distinction between `FAILED`, `STALE_DATA`, and `SKIPPED` is important. Stale or unavailable data does not automatically indicate a software failure.

## Conditional Retraining

Model retraining is conditional rather than automatic.

The retraining decision can instruct AeroAI to:

- wait for additional labelled observations,
- continue monitoring and collecting data, or
- allow candidate model training when the required conditions are satisfied.

A newly trained candidate does not automatically replace the active model. Candidate validation is performed before model promotion, and the existing model remains unchanged when the validation requirements are not satisfied.

## Fault-Tolerant Execution

AeroAI is designed so that a problem with one data source does not unnecessarily stop independent parts of the system.

For example, if the original historical source becomes stale, its dependent prediction or monitoring stages can be skipped while the independent OpenAQ monitoring workflow continues to process fresh OpenAQ observations.

This allows the complete pipeline to report `PARTIAL_SUCCESS` instead of incorrectly treating every external data-source problem as a total system failure.

---

# Machine Learning Model Development and Validation

## Purpose of Machine Learning

The machine-learning component of AeroAI is used for next-day AQI forecasting. AQI calculation and AQI prediction are treated as different operations. AQI calculation applies pollutant breakpoint rules to measured observations, whereas machine learning attempts to estimate a future AQI value from historical information.

## Prediction Inputs and Target

The forecasting workflow uses air-quality and environmental features prepared from historical observations. The project includes pollutant concentrations, temperature, humidity and pollutant AQI-related features where applicable.

The prediction target represents the next-day AQI value associated with the corresponding modelling workflow. Target provenance is retained so that a legacy estimated-AQI model is not incorrectly presented as an official CPCB AQI forecasting model.

## Models Evaluated

AeroAI evaluates tree-based regression models including:

- Random Forest
- XGBoost

Multiple candidate models are compared instead of assuming that one algorithm will always provide the best forecast.

## Evaluation Metrics

Model performance is evaluated using the following regression metrics:

### Mean Absolute Error (MAE)

MAE represents the average absolute difference between predicted and observed AQI values. Lower MAE indicates smaller average prediction error.

### Root Mean Squared Error (RMSE)

RMSE also measures prediction error but gives greater weight to large individual errors. Lower RMSE is preferred.

### R-squared (R2)

R2 describes how much of the variation in the target is explained by the model. It is used together with error metrics rather than as the only model-selection criterion.

## Persistence Baseline

Experimental forecasting models are compared with a persistence baseline.

The persistence baseline assumes that the next AQI value will remain similar to the most recent available AQI value. This provides a simple reference against which the usefulness of a more complex machine-learning model can be evaluated.

A candidate model should demonstrate meaningful and sufficiently consistent improvement over the baseline before production promotion is considered.

## Chronological Validation

Air-quality forecasting is a time-dependent problem. AeroAI therefore avoids randomly mixing future observations into the training data when evaluating future predictions.

Chronological validation keeps earlier observations in the training period and later observations in the evaluation period.

This helps reduce information leakage and provides a more realistic estimate of how the model may behave when forecasting unseen future observations.

## Walk-Forward Validation

The OpenAQ experimental workflow includes expanding-window walk-forward validation.

In this approach:

1. The model is trained using an initial historical period.
2. It is evaluated on a later period that was not used for training.
3. The training window is expanded with additional historical observations.
4. The model is evaluated again on the next future period.
5. Performance is examined across multiple chronological folds.

This approach tests whether candidate performance is reasonably consistent across different future periods rather than relying on a single train-test split.

## Trend-Aware Experiment

AeroAI also includes an experimental trend-aware forecasting workflow.

The purpose of this experiment is to examine whether recent AQI history, lag information, short-term trends and calendar information can improve next-day forecasting compared with simpler input features.

Lag features are used only when the required previous calendar observations are available. Missing historical dates are not silently treated as continuous observations.

The trend-aware experiment remains an evaluation workflow and does not automatically promote a trained model.

## Candidate Model Promotion

Candidate training and production model promotion are separate operations.

A candidate model may be trained and evaluated without replacing the active model. Promotion requires the candidate to satisfy the configured validation requirements, including comparison with a baseline and consistency across chronological evaluation periods.

If these requirements are not satisfied, the candidate remains experimental and the existing champion model remains unchanged.

## Model Safety

AeroAI applies the following safeguards to the model lifecycle:

- model targets are explicitly identified;
- feature compatibility is checked before prediction;
- stale observations can block current forecasting;
- candidate models are evaluated before promotion;
- baseline comparison is included in experimental validation;
- chronological evaluation is preferred for time-dependent forecasting; and
- experimental OpenAQ-derived AQI targets are not presented as official CPCB AQI.

These safeguards are intended to prevent a technically successful training run from being interpreted as sufficient evidence for production deployment.
