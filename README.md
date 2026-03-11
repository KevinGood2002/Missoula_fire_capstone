# Missoula Fire Dispatch Analysis

## Project Overview
This project analyzes emergency dispatch data from a Missoula fire station to better understand incident response performance. The analysis focuses on identifying patterns in dispatch and turnout times and exploring factors that may contribute to delays.

The project is part of a broader effort to improve operational awareness through data analytics and visualization.

## Data Pipeline
Dispatch data is processed and automated using Microsoft tools:

- **Power Automate** is used to collect and process incoming dispatch data.
- Processed data is stored and prepared for reporting.
- **Power BI** is used to build a dashboard that visualizes turnout times and other key response metrics.

The goal of this pipeline is to create a repeatable workflow that updates dashboards automatically as new data becomes available.

## Repository Structure

Missoula_fire_capstone/
│
├── data/
│ Raw or cleaned dispatch datasets used in the analysis
│
├── python_scripts/
│ Python scripts used for data cleaning, exploratory analysis, and modeling
│
├── outputs/
│ Generated results such as summary tables, figures, and processed datasets
│
└── README.md

## Key Focus Areas

- Dispatch time analysis
- Turnout time performance
- Identification of operational trends in fire response data
- Data automation and dashboard integration

## Tools Used

- Python (pandas, numpy, matplotlib, statsmodels)
- Microsoft Power Automate
- Microsoft Power BI

## Author

Kevin Good  
University of Montana
