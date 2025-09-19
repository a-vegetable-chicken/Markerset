# Skin-Mounted vs. Shoe-Mounted Markers
### Code for the paper:
**Skin-Mounted vs. Shoe-Mounted Markers: Is There a Significant Difference in Measuring Multi-Segment Foot Kinematics with Lateral Wedge Insoles?**

---

## 📖 Introduction
This repository contains the code accompanying our paper *"Skin-Mounted vs. Shoe-Mounted Markers: Is There a Significant Difference in Measuring Multi-Segment Foot Kinematics with Lateral Wedge Insoles?"*.  

---

## ⚙️ Requirements
This project was developed and tested with **Python 3.13**.  
To replicate the environment used in our experiments, please export your current environment and share it as `requirements.txt` (for pip) or `environment.yml` (for conda).  

### Using pip
```bash
# Export current environment
pip freeze > requirements.txt

# Recreate environment
pip install -r requirements.txt

---

### Using Conda

# Export current environment
conda env export --no-builds > environment.yml

# Recreate environment
conda env create -f environment.yml

---

## 🚀 Usage
1. Clone the repository:  
   `git clone git@github.com:a-vegetable-chicken/Markerset.git`  
   `cd Markerset`  

2. Run the preprocessing pipeline:  
   `python preprocess.py`  

3. Run the statistical analysis:  
   `python analysis.py`  

4. (Optional) Generate figures:  
   `python plot_results.py`  

---
