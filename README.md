
# 🚴 Cycling Kinematic Analysis Tool

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)
![Status](https://img.shields.io/badge/status-research--prototype-yellow)

Python tool for the analysis of lower limb 3D kinematics during cycling across multiple power levels.  
Designed for motion capture recordings from Contemplas™ or similar systems.

---

## 📦 Features

- Supports multiple `.txt` motion capture files per subject
- Range of Motion (ROM), Symmetry Index (NSI), and Cross-Correlation analysis
- Per-joint, per-DOF asymmetry quantification
- Repeated Measures ANOVA or Friedman test (auto-selected based on normality)
- Bonferroni-corrected pairwise t-tests
- CSV summary + diagnostic plots (optional PDF or Excel integration)

---

## 🖥️ Installation

### 🔹 Requirements

- Python 3.9 or higher (tested with Python 3.10)

### 🔹 Setup Instructions

1. Clone this repository or download the `.py` file:

   ```bash
   git clone https://github.com/yourusername/cycling-kinematics.git
   cd cycling-kinematics
   ```

2. Install required packages:

   ```bash
   pip install pandas numpy matplotlib scipy statsmodels
   ```

---

## ▶️ Usage

1. Run the script:

   ```bash
   python multi_cycling_kinematic_analysis_FINAL.py
   ```

2. Follow the GUI prompts:
   - Select a `.txt` motion file (one per power level)
   - Input the associated power level (e.g. 290)
   - Repeat until all desired files are selected
   - Click "Finish and Start Analysis"

---

## 📁 Outputs

For each subject/session:

1. `Combined_Kinematic_Results.csv`  
   → Contains one row per joint/DOF/power level, including:
   - ROM, NSI metrics, r_max, τ_lag
   - Min/max angles
   - Repeated Measures statistics: mean, SD, ANOVA/Friedman, pairwise tests

2. `/Comparison_Plots/`  
   → Contains PNG plots showing each metric vs power level (e.g. NSI_abs vs Power for L_HIP Flex/Ext)

---

## 📖 Methods
read "method.txt" for a more detalied explanation. 
### Implemented Calculations:

1. **Range of Motion (ROM)** per joint and axis  
2. **Cross-correlation** between left/right time series  
3. **Symmetry Index (NSI)**:  
   - min-max normalized  
   - mean + absolute mean NSI curves
4. **Shapiro-Wilk test** for normality  
5. **ANOVA or Friedman** test (auto-selected)  
6. **Bonferroni-corrected** pairwise t-tests

---

## 📜 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.  
Use for academic and research purposes is permitted with proper attribution.  
Commercial use is prohibited without explicit permission.

📧 Contact for licensing: ophiravina@gmail.com

---

## 🧠 Author

Ophir Ravina  
📧 ophiravina@gmail.com

---

## 🙏 Citation

If you use this tool in your research, please cite:

> Ravina, O. (2025). Cycling Kinematic Analysis Tool (Version 1.0.0) [Computer software].  
> Zenodo. https://doi.org/10.xxxx/zenodo.xxxxxxx
