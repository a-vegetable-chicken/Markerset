# Markerset

Code for the paper:  
Skin-Mounted vs. Shoe-Mounted Markers: Is There a Significant Difference in Measuring Multi-Segment Foot Kinematics with Lateral Wedge Insoles?  

This repository contains code and analysis pipelines for a biomechanics research project, including preprocessing of motion capture data, multi-segment foot kinematic analysis, application of lateral wedge insoles, and statistical modeling (e.g., ANOVA, ANCOVA).  

---

## 🔧 Getting Started

### 📥 Clone the repository
Use SSH to clone this repository from GitHub:  
```bash
git clone git@github.com:a-vegetable-chicken/Markerset.git
cd Markerset
```

---

### 📦 Set up the environment
This project was developed and tested with **Python 3.13**.  

#### Using pip
```bash
pip freeze > requirements.txt
pip install -r requirements.txt
```

#### Using conda
```bash
conda env export --no-builds > environment.yml
conda env create -f environment.yml
conda activate <env_name>
```

---

### ▶️ Run the main scripts
Run preprocessing pipeline:  
```bash
python preprocess.py
```  
Run the statistical analysis:  
```bash
python analysis.py
```  

### Generate figures (optional):  
```bash
python plot_results.py
```


---

## 📄 Citation
If you use this code, please cite our paper:   
```bibtex
@article{your_bibtex_key,
   title={Skin-Mounted vs. Shoe-Mounted Markers: Is There a Significant Difference in Measuring Multi-Segment Foot Kinematics with Lateral Wedge Insoles?},
  author={Anonymous},
  journal={Submitted to [Journal/Conference]},
   year={2025}
 }
```

---

## 👥 Authors
- Anonymous for double-blind review  

---

## 📄 License
This code is released for academic and review purposes only.  
