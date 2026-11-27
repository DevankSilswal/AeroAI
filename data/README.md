# Data directory

Runtime datasets and generated reports are intentionally not committed.

AeroAI can retrieve data from:
- XKDR Air Quality API
- OpenAQ API v3
- CPCB monitoring data published through data.gov.in
- Open-Meteo weather API

Create the required local data subdirectories and populate them using
the collection scripts documented in the root README.

Do not redistribute third-party datasets unless their terms and
redistribution permissions have been checked. API credentials belong
in the local `.env` file, never in Git.
