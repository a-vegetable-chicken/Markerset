# LMB

Laboratory for Movement Biomechanics (LMB) Project Repository

This repository contains code and data processing pipelines for a biomechanics research project
---

## 🔧 Getting Started

### 📥 Clone the repository

Use SSH to clone this repository from ETH GitLab:

```bash
git clone git@gitlab.ethz.ch:qizhang/lmb_innosuisse_shoe.git
cd lmb_innosuisse_shoe
```

> You must have ETH GitLab access and SSH keys configured.

---

### 📦 Set up the conda environment

This project uses [conda](https://docs.conda.io/en/latest/miniconda.html) for dependency management.

```bash
conda activate lmb
```

- Python version: **3.13**
- Main dependencies include:
  - `numpy`, `pandas`, `matplotlib`, `openpyxl`
  - `scikit-learn`, `seaborn`, `xlrd`, `trimesh`
- Full list of dependencies is available in [`environment.yml`](./environment.yml)

---

### ▶️ Run the main script

To execute the main module (e.g., 3D reconstruction or animation):

```bash
cd knee_coord
python main.py
```

To run specific analysis modules:

```bash
# Example: Plots generation
python plot.py

# Example: Regression analysis
python regression_analysis.py
```

Ensure required input files (e.g., `.stl` meshes, Excel data) are placed in the correct folders as expected by the scripts or config files.

---

## 📁 Code Structure

### 1. Dual-plane Fluoroscope

This module processes dual-plane fluoroscopic, dealing with kinetic and kinematic data to reconstruct bone motion and analyze joint contact.
#### 1.1 Data Preprocessing
- dual-camera system setting up

- Kinematic and Kinetic data integration and alignment

#### 1.2 Model Reconstruction
- 3D bone model reconstruction (e.g., femur, tibia)

- Contact depth and penetration region estimation

#### 1.3 Visualization
- Rendering of aligned bone models

- Export of animations and video sequences for visualization

---
### 2. 3D Motion Capture (Under Development)

#### 2.1 Data Preprocessing
- Parse raw outputs data exported from V3D and automatically identify valid entries and reformat multi-line headers
- Organize extracted biomechanical parameters by subject and condition
- Export structured data into Excel files for downstream analysis, with consistent formatting and dimension handling

---
### 3. Plantar Force and CoP Analysis

This Module is for the preprocessing and analysis of plantar force data collected from both sensor insoles and force plates. Key steps include data loading and cleaning, temporal normalization across trials, and visualization of center of pressure (CoP) and related dynamics.

#### 3.1 Data Structuring and Cleaning
- Load and organize raw plantar force data by subject, condition, and trial..
- Convert raw data into structured arrays aligned by gait events or step cycles.

#### 3.2 Temporal Alignment 
- Align plantar force data with external event markers.
- Normalize the duration of gait cycles to enable inter-trial and inter-subject comparisons.

#### 3.3 Descriptive Plotting and Exploratory Analysis
- Visualize biomechanical parameters within normalized individual gait cycles.
- Visualize CoP trajectories and analyze pressure distribution.
- Explore spatial-temporal patterns through condition-wise comparisons.

---
### 4. Data Integration and Modeling
This module is for the integration of multi-modal data and the development of Artificial Intelligence models to estimate key biomechanical parameters.

#### 4.1 Unified Data Structures
- Merge multi-modal data (e.g., insole + MoCap).
- Normalize and temporally align signals.

#### 4.2 Biomechanical Feature Engineering and Matching
- Derive interpretable biomechanical variables (e.g., CoP trajectory, GRF profiles, joint kinetics).
- Construct temporally aligned feature sets for downstream modeling.

#### 4.3 Statistical Modeling
- Implement multi-linear regression and related parametric models .
- Evaluate model performance using established statistical metrics and validation protocols.

---

## 📌 TODO

### 🧩 1. Refactor and Simplify Parameter Setting
- [ ]Centralize all parameters into a shared configuration file to facilitate reuse and modification by other users

### 🎥 2. Continue MoCap Module Development 
- [ ] Develop the MoCap module to extract clean parameters, detect gait events, and align with other data sources.
### 🤖 3. Apply Machine Learning for Multimodal Analysis
- [ ] Design multimodal input pipeline (e.g., insole + MoCap + GRF)  
- [ ] Explore baseline models (e.g., Random Forest, MLP)  
 

## 👥 Authors

- **Tingyu Wang**  
  Laboratory for Movement Biomechanics, ETH Zürich  
  [wangtingyu@student.ethz.ch](mailto:wangtingyu@student.ethz.ch)

## 📄 License

This project is intended for internal academic use. For reuse or distribution, please contact the authors or project PI.
