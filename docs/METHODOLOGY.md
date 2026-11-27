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
