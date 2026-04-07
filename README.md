🚀 EOBDS: Constant-Time Inference for Binary Decision Trees
📌 Overview

Binary Decision Trees (BDTs) are widely used in machine learning due to their interpretability and effectiveness. However, their inference time is inherently depth-dependent and stochastic, which makes them inefficient for real-time and fixed-rate streaming applications.

This repository presents a novel framework that reformulates BDTs into Boolean decision representations, enabling constant-time inference.

🧠 Key Idea

Traditional BDT inference depends on:

Tree depth
Path traversal
Input-dependent branching

👉 This leads to variable latency

✅ Proposed Solution

We transform BDTs into Boolean functions and optimize them through multiple stages:

BDS (Boolean Decision System)
Converts BDT into Boolean representation
Eliminates traversal dependency
OBDS (Optimized BDS)
Merges nodes with approximate boundary similarity
Reduces redundancy
EOBDS (Espresso-Optimized BDS)
Applies the ESPRESSO logic minimization algorithm
Produces compact and efficient Boolean expressions

⚡ Key Contributions
✅ Converts BDTs and Random Forests into Boolean representations
✅ Achieves constant-time inference (independent of tree depth)
✅ Reduces computational complexity for streaming systems
✅ Uses ESPRESSO optimization for logic minimization
✅ Maintains statistical equivalence with original models

📊 Performance Insights
Inference time: O(1)
Works for:
Binary Decision Trees (BDTs)
Random Forests (RFs)
Empirical results show:
No significant loss in accuracy
Consistent latency across inputs

EOBDS/
│
├── TrainingFiles/        # Input datasets (ignored in Git)
├── Output/               # Generated results
│
├── basic_functions.py
├── bds_fun.py            # BDS construction
├── new_bds_fun.py
├── new_obds_func.py      # OBDS implementation
├── new_train.py          # Training pipeline
├── predictf.py           # Inference module
│
├── IG_func.py            # Information gain
├── gmm_mml.py            # Model utilities
├── MintermCal.py         # Boolean minterm calculations
│
└── .gitignore

⚙️ Installation
🔹 Create virtual environment

python -m venv venv
venv\Scripts\activate

pip install numpy pandas matplotlib scipy scikit-learn pyeda

▶️ Usage
🔹 Train Model
python new_train.py
🔹 Run Inference
python predictf.py

🔬 Methodology
Train BDT / Random Forest
Convert decision paths → Boolean expressions
Apply:
Node aggregation (OBDS)
Logic minimization (EOBDS)
Perform inference using optimized Boolean function

📈 Applications
Real-time inference systems
Edge AI / embedded systems
Hardware acceleration (FPGA / ASIC)
Streaming data pipelines

👨‍💻 Author

M. Srinivas
PhD Student, Electrical Engineering,
Indian Institute of Technology Kharagpur
Dr D. Sheet
Associate Professor, Electrical Engineering,
Indian Institute of Technology Kharagpur
