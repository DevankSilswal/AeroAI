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
