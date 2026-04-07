# 🚀 EOBDS: Constant-Time Inference for Binary Decision Trees

## 📌 Overview

Binary Decision Trees (BDTs) are widely used in machine learning due to their interpretability and effectiveness. However, their inference time is inherently **depth-dependent and stochastic**, making them inefficient for **real-time and fixed-rate streaming applications**.

This repository introduces a novel framework that reformulates BDTs into **Boolean decision representations**, enabling **constant-time inference** independent of tree depth.

---

## 🧠 Key Idea

Traditional BDT inference depends on:

* Tree depth
* Path traversal
* Input-dependent branching

👉 This leads to **variable latency and unpredictable inference time**

---

## ✅ Proposed Solution

We transform BDTs into Boolean functions and optimize them through the following stages:

### 🔹 BDS (Boolean Decision System)

* Converts BDT into Boolean representation
* Eliminates traversal dependency

### 🔹 OBDS (Optimized BDS)

* Aggregates nodes with **approximate boundary similarity**
* Reduces redundancy in representation

### 🔹 EOBDS (Espresso-Optimized BDS)

* Applies the **ESPRESSO logic minimization algorithm**
* Produces compact and efficient Boolean expressions

---

## ⚡ Key Contributions

* ✅ Converts BDTs and Random Forests into Boolean representations
* ✅ Achieves **constant-time inference (O(1))**
* ✅ Eliminates dependency on tree depth
* ✅ Reduces computational complexity for streaming systems
* ✅ Applies **ESPRESSO logic minimization**
* ✅ Maintains **statistical equivalence** with original models

---

## 📊 Performance Insights

* ⏱️ **Inference Complexity:** O(1)
* 📌 Applicable to:

  * Binary Decision Trees (BDTs)
  * Random Forests (RFs)

### Empirical Observations:

* ✔ No significant loss in accuracy
* ✔ Uniform inference latency
* ✔ Scalable to multiple trees

---

## 🏗️ Project Structure

```
EOBDS/
│
├── TrainingFiles/        # Input datasets (ignored in Git)
├── Output/               # Generated outputs/results
│
├── basic_functions.py
├── bds_fun.py            # BDS construction
├── new_bds_fun.py
├── new_obds_func.py      # OBDS implementation
├── new_train.py          # Training pipeline
├── predictf.py           # Inference module
│
├── IG_func.py            # Information gain computation
├── gmm_mml.py            # Model utilities
├── MintermCal.py         # Boolean minterm computation
│
└── .gitignore
```

---

## ⚙️ Installation

### 🔹 Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

### 🔹 Install Dependencies

```
pip install numpy pandas matplotlib scipy scikit-learn pyeda
```

---

## ▶️ Usage

### 🔹 Train Model

```
python new_train.py
```

### 🔹 Run Inference

```
python predictf.py
```

---

## 🔬 Methodology

1. Train a BDT or Random Forest
2. Extract decision paths
3. Convert paths → Boolean expressions
4. Apply:

   * Node aggregation (OBDS)
   * Logic minimization (EOBDS using ESPRESSO)
5. Perform inference using optimized Boolean logic

---

## 📈 Applications

* Real-time inference systems
* Edge AI and embedded systems
* Hardware acceleration (FPGA / ASIC)
* Streaming data pipelines
* Low-latency decision systems

---

## ⚠️ Notes

* Large datasets are excluded via `.gitignore`
* Ensure `pyeda` is installed for ESPRESSO optimization
* Recommended to use a virtual environment


## 👨‍💻 Authors

**M. Srinivas**
PhD Student, Electrical Engineering
Indian Institute of Technology Kharagpur

**Dr. D. Sheet**
Associate Professor, Electrical Engineering
Indian Institute of Technology Kharagpur

---

## ⭐ Future Work

* Hardware implementation of EOBDS (FPGA/ASIC)
* Extension to multi-class classification
* Integration with deep learning pipelines
* Optimization for large-scale forests

---

## 🏁 Summary

This work transforms traditional tree-based models into **hardware-friendly, constant-time inference systems**, making them highly suitable for **real-time, latency-critical applications**.
