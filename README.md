# 🐠 Coral Reef Monitoring in the Florida Keys

## 📘 Project Overview

This project analyzes coral reef health trends within the **Florida Keys National Marine Sanctuary**, using historical monitoring data from the **Coral Reef Evaluation and Monitoring Project (CREMP)**. The analysis focuses on **stony corals**, **octocorals**, and **environmental parameters like temperature**, aiming to uncover trends, correlations, and actionable insights for reef conservation.

---

## 🎯 Objectives

- **Analyze** long-term trends in **stony coral percent cover** and **species richness**
- **Compare** reef health and biodiversity **across stations and subregions**
- **Correlate** temperature with coral health indicators
- **Forecast** future coral reef status using predictive models
- **Identify** key influencing species and early warning signs of reef degradation

---

## 🧾 Dataset Sources

- `CREMP_Pcover_2023_StonyCoralSpecies.csv`  
- `CREMP_SCOR_Summaries_2023_Counts.csv`  
- `CREMP_Temperatures_2023.csv`  
- `CREMP_OCTO_Summaries_2023_Density.csv`

Datasets were sourced from CREMP under the EPA’s Water Quality Protection Program.

---

## 🛠️ Technologies Used

- **Python** (Pandas, Matplotlib, Seaborn, Scikit-learn)
- **Statistical Analysis** (Correlation, Linear Regression)
- **Data Visualization** (Trends, Correlations, Distributions)
- **PDF Report Generation** (`fpdf`)

---

## 📊 Key Visualizations

- 📉 *Stony Coral Percent Cover Over Time*
- 🌱 *Species Richness Trends*
- 🔥 *Coral Cover vs Temperature*
- 📍 *Subregion-Wise Coral Health Comparisons*
- 🧮 *Forecast: Coral Cover in Next 10 Years*
- 🧬 *Octocoral Density Patterns*
- 🧠 *Heatmap of Influencing Coral Species*

_All visuals are exported in the `/outputs` directory and used in the final report._

---

## 📄 Final Report

A comprehensive, non-technical [PDF Report](./Florida_Keys_Coral_Report.pdf) is included to explain findings to stakeholders, policy-makers, and non-data experts.

---

## 🧠 Insights & Recommendations

- **Temperature Rise** significantly impacts coral cover—monitoring is essential.
- Some subregions (e.g., **UK**) show resilience—prioritize conservation here.
- **Species diversity** has held steady, a positive sign for long-term recovery potential.
- Early declines in specific coral species may act as warning signals for broader degradation.

---

## 📁 Directory Structure
## ✅ How to Run

1. Clone this repository  
2. Place all CSV datasets inside the `/CREMP_CSV_files` directory  
3. Run `Data_challenge.py`  
4. Plots will be saved to `/outputs` and `Florida_Keys_Coral_Report.pdf` will be generated

---

## 🙌 Acknowledgments

Thanks to the **Fish and Wildlife Research Institute**, **EPA**, and **CREMP** for data access. This analysis was developed as part of the **OpenMiami Datathon** to support resilience and sustainability through open data.

---

## 📬 Contact

**Sahil Srivastava**  
Student | Data Science Enthusiast  
📧 sahilsrivastava773@gmail.com  

---
