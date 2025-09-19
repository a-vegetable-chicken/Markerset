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

#### Standing analysis
Run ANCOVA:
```bash
python Standing.py --analysis ancova
```

Run Panel3:
```bash
python Standing.py --analysis panel3
```

---

#### Walking analysis
Run paired t-tests:
```bash
python Walking.py --analysis paired
```

Run RM-ANOVA with post-hoc:
```bash
python Walking.py --analysis anova
```

Run gait condition means ± SD plots:
```bash
python Walking.py --analysis plot
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
