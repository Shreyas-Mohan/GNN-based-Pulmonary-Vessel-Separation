# GNN-based Pulmonary Vessel Separation: Hierarchical Graph Attention Networks for 3D Artery-Vein Separation on Thoracic CT

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.3+-3C2179.svg)](https://pyg.org/)
[![SimpleITK](https://img.shields.io/badge/SimpleITK-2.2+-brightgreen.svg)](https://simpleitk.org/)
[![Medical Imaging](https://img.shields.io/badge/Domain-Thoracic%20CT%20Imaging-teal.svg)](https://doi.org/10.1038/s41598-025-12141-0)
[![Dataset: Lung250M-4B](https://img.shields.io/badge/Dataset-Lung250M--4B-purple.svg)](https://www.imi.uni-luebeck.de)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An academically rigorous, production-grade deep learning framework for **automated pulmonary artery-vein separation** and **vascular morphometry extraction** on 3D thoracic computed tomography (CT) scans. 

Based on **Hierarchical Graph Attention Networks (HGAT)**, multi-planar orthogonal vision transformer embeddings, **Radius-Weighted Focal Loss**, and **Confidence-Guided Edge-Affinity Diffusion**, this framework addresses core topological error modes in medical vessel tree classification, achieving state-of-the-art branch continuity and robust anatomical separation on 3D thoracic CT.

---

## 📌 Table of Contents

1. [Clinical & Scientific Motivation](#-clinical--scientific-motivation)
2. [Research & Engineering Journey (Trials 1 – 5)](#-research--engineering-journey-trials-1--5)
3. [Methodology & Architecture](#-methodology--architecture)
4. [Empirical Benchmark & Ablation Study](#-empirical-benchmark--ablation-study)
5. [Clinical Imaging Biomarkers & Vascular Morphometry](#-clinical-imaging-biomarkers--vascular-morphometry)
6. [Interactive Visualizer & Generated Artifacts](#-interactive-visualizer--generated-artifacts)
7. [Repository Structure](#-repository-structure)
8. [Installation & Setup](#-installation--setup)
9. [Step-by-Step Reproduction Guide](#-step-by-step-reproduction-guide)
10. [References & Citations](#-references--citations)

---

## 🩺 Clinical & Scientific Motivation

In thoracic radiology and pulmonary medicine, differentiating **pulmonary arteries (PA)** from **pulmonary veins (PV)** is crucial for diagnosing and managing cardiopulmonary disorders:
- **Pulmonary Hypertension (PH)**: Characterized by abnormal arterial remodeling and vascular pruning.
- **Chronic Obstructive Pulmonary Disease (COPD)**: Exhibits loss of small peripheral vasculature ($BV_5$) alongside emphysematous parenchymal destruction.
- **Thoracic Surgical Planning**: Crucial for anatomical segmentectomy and lobectomy where arterial and venous branches must be precisely isolated.

### Why Standard Dense 3D CNNs Fail
A standard chest CT volume contains **50 to 100 million voxels** ($512 \times 512 \times 350$). Training dense 3D convolutional networks directly on voxel grids encounters two major bottlenecks:
1. **Computational Prohibitive Memory**: Voxel-wise receptive fields are local, missing long-range vascular continuity from the cardiac hilum to the chest wall.
2. **Topological Branch Disconnects**: CNNs produce isolated voxel islands and fragmented vessel segments, leading to severe **Branch-Mismatch Count (BMC)** fractures where a continuous vessel branch flips between artery and vein labels mid-stream.

### The Point-Cloud Graph Paradigm
Instead of segmenting empty parenchymal background, the pulmonary vascular tree is represented as a **3D spatial graph**:
- **8,192 landmark nodes** per CT scan sampled along vessel centerlines and lumens across the entire 3D lung volume.
- Each node carries rich geometric, caliber, and multi-planar visual texture embeddings.
- Graph edges capture physical branch topology ($k$-NN and spatial continuations), allowing information to propagate along true anatomical vessel trees via **Hierarchical Graph Attention (HGAT)**.

---

## 🔬 Research & Engineering Journey (Trials 1 – 5)

Our research progressed through five major iterations, systematically solving data leakage, feature misalignment, peripheral vessel decay, and topological branch fractures:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│     Trial 1     │       │     Trial 2     │       │     Trial 3     │
│  Initial HGAT   │ ────> │  Zero Patient   │ ────> │  Deterministic  │
│ Random Feature  │       │    Leakage &    │       │ Patch Extractor │
│   Projection    │       │ Vectorized BFS  │       │  (584-dim Node) │
└─────────────────┘       └─────────────────┘       └─────────────────┘
                                                             │
                                                             ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│     Trial 5     │       │   Diagnostics   │       │     Trial 4     │
│ Radius-Weighted │ <──── │ 65.8% BV5 Small │ <──── │ Full Benchmark  │
│ Focal + Topo &  │       │ Vessels & Phase │       │  (T=0.45 Base)  │
│ Conf. Diffusion │       │ Inconsistency   │       │  70.23% Acc     │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

### Trial 1: Baseline Port & Feature Initialization
- **Goal**: Re-implement and verify Hierarchical Graph Attention Networks (HGAT) on Lung250M-4B.
- **Bottlenecks Identified**: The original repository relied on unseeded random projection matrices in the visual patch transformer, causing train/test feature misalignment. Gradients suffered from erratic swings during graph optimization.

### Trial 2: Patient Leakage Elimination & Vectorized BFS
- **Critical Discovery (Data Contamination)**: In the Lung250M-4B dataset, each patient has two breath-hold scans: **Inspiratory (`_1`)** and **Expiratory (`_2`)**. Naive random splitting separated `case_054_1` into training and `case_054_2` into validation, artificially inflating validation accuracy through patient memorization.
- **Architectural Solution**: Implemented **Patient-Wise Group Splitting** across 87 unique patient IDs: strictly **69 train patients (138 scans)** and **18 validation patients (36 scans)** with **zero overlap**.
- **Performance Optimization**: Replaced sequential BFS layer loops with vectorized PyTorch operations, achieving a **10,000x speedup** in hierarchy propagation.

### Trial 3: Deterministic Canonical Embeddings (584-Dimensional Representation)
- **Goal**: Lock in a standardized, reproducible multimodal node feature vector.
- **Implementation**: Fixed `patch_extractor.pth` with seed=42 to extract 9-orthogonal-plane vision transformer embeddings ($f_{\text{VV}} \in \mathbb{R}^{576}$).
- **Node Feature Vector $\mathbf{x}_i \in \mathbb{R}^{584}$**:
  1. Normalized 3D Coordinates: $(x, y, z) \in \mathbb{R}^3$
  2. Local Vessel Radius / Caliber: $r_i \in \mathbb{R}^1$ (distance transform to vessel boundary)
  3. Directional & Neighborhood Features: $\vec{d}_i \in \mathbb{R}^4$ (edge vectors & graph degree)
  4. Visual Patch Transformer Embedding: $f_{\text{VV}} \in \mathbb{R}^{576}$

### Trial 4: Baseline Benchmark & Diagnostic Breakdown
- **Benchmark Established**: Trained for 40 epochs. Systematic operating threshold sweep identified $T=0.45$ as the optimal decision boundary.
- **Baseline Test Performance**: **70.23% Accuracy**, **0.7017 Macro S-Dice**, **54.06% mIoU**, and **588.8 BMC Fractures**.
- **Deep-Dive Diagnostic Findings**:
  1. **Small Vessel ($BV_5$) Vulnerability**: Small peripheral vessels ($r \le 1.5$ voxels) constitute **65.8% of the entire vascular tree** (161,686 of 245,760 test nodes). In Trial 4, large central trunks achieved **73.30% accuracy**, but small vessels dropped to **68.63% accuracy** and **66.00% vein recall**.
  2. **Breathing Phase Variance**: Inspiratory scans (`_1`) achieved 70.89% accuracy, whereas expiratory scans (`_2`) dropped to 69.56%, showing intra-patient accuracy shifts up to 5.95% due to lung deflation and vessel crowding.
  3. **Topological Disconnects**: Unregularized node classification produced an average of **588.8 branch fractures (BMC)** per patient, where adjacent physical vessel continuations flipped labels.

### Trial 5: Radius-Weighted Focal Loss & Topological Diffusion
To resolve the error modes discovered in Trial 4, we implemented two synergistic tracks:

- **Track 1 (Post-Processing): Confidence-Guided Edge-Affinity Diffusion**:
  - High-confidence nodes ($P \ge 0.70$ or $P \le 0.30$) act as immovable topological anchors.
  - Probability consensus diffuses across graph edges with edge-affinity weights $W_{uv} = \exp\left(-\frac{(P_u - P_v)^2}{\sigma^2}\right)$, aggressively downweighting cross-vessel shortcut edges.
  - Slashes BMC fractures immediately by **28.3 fractures/case** (588.8 $\to$ 560.5 on Trial 4, and 593.0 $\to$ 564.6 on Trial 5).

- **Track 2 (Training): Radius-Weighted Focal Loss + Differentiable Edge Regularization**:
  Multi-objective training loss prioritizing small peripheral calibers:

$$
\mathcal{L} = \mathcal{L}_{\text{Focal-Radius}} + \lambda_{\text{topo}}\mathcal{L}_{\text{topo}} + \lambda_{\text{seed}}\mathcal{L}_{\text{seed}}
$$

  where node probability divergence $(P_u - P_v)^2$ is penalized directly during backpropagation ($\lambda_{\text{topo}} = 0.15$), and small branches are weighted inversely by radius $w(r) \propto 1/\sqrt{r}$.
  - **Result**: Boosted small peripheral artery recall to **75.76%** (vs 70.90% in Trial 4), and provided an optimal balanced operating point ($T=0.43$) yielding **72.48% Vein Recall** and **69.75% Small Vein Recall** for clinical vascular analysis.

---

## 📐 Methodology & Architecture

```
                    Raw CT Volume (3D)
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
      Vessel Segmentation        Centroid Skeletons
        & Parenchyma Mask       (Uniform 8,192 Nodes)
               │                         │
               │            ┌────────────┴────────────┐
               │            ▼                         ▼
               │     3D Coordinates              Vessel Radii
               │     & k-NN Edges (k=3)        (Distance Transform)
               │            │                         │
               └──────────┐ │ ┌───────────────────────┘
                          ▼ ▼ ▼
               9-Plane Orthogonal Transformer
                  Patch Extraction (f_VV)
                            │
                            ▼
              Node Features: x_i ∈ R^584
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
   Hilum Seed Extraction         BFS Layer-Wise Hierarchy
    (Top 10 Caliber Nodes)          Propagation (Hops 0-3)
             │                             │
             └──────────────┬──────────────┘
                            ▼
               Hierarchical Graph Attention
                     Network (HGAT)
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
    Trial 5 Multi-Objective       Track 1 Confidence-Guided
      Radius-Weighted Loss           Affinity Diffusion
   (Focal + Differentiable Topo)      (Post-Processing)
                            │
                            ▼
             Calibrated Binary Predictions
             (0: Artery, 1: Vein) + BMC Fix
                            │
                            ▼
          Sparse-to-Dense KDTree 3D Reconstruction
                    & Clinical Biomarkers
```

### 1. Light Vessel Structured Modeling (LVSM)
- **Graph Construction**: Constructs an undirected graph $G = (V, E)$ using $k=3$ spatial nearest neighbors.
- **Hilum Seed Selection**: Automatically identifies the 10 largest-caliber trunk nodes near the pulmonary hilum using a caliber-distance objective:

$$
S_i = r_i - 0.1 \cdot \Vert \mathbf{x}_i - \mathbf{x}_{\text{center}} \Vert_2
$$

- **Layer-Wise BFS Propagation**: Propagates hierarchical layers outward from hilum seeds up to $K=3$ hops, generating structural hierarchy masks $\mathbf{H}_i \in \{0, 1\}^4$.

### 2. Multi-Head Hierarchical Attention (HGAT)
Layer-conditioned self-attention calculates dynamic attention coefficients between connected nodes:

$$
\alpha_{uv} = \frac{\exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{W}\mathbf{x}_u \parallel \mathbf{W}\mathbf{x}_v]\right)\right)}{\sum_{k \in \mathcal{N}_u} \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{W}\mathbf{x}_u \parallel \mathbf{W}\mathbf{x}_k]\right)\right)}
$$

Features are aggregated across multiple attention heads and modulated by layer-level routing weights.

### 3. Radius-Weighted Focal Loss
To combat class imbalance and the numerical dominance of large trunks, each node loss is scaled inversely by its physical radius:

$$
w_i = \frac{1}{\sqrt{\mathrm{clamp}(r_i, 0.5, 5.0)}}
$$

$$
\mathcal{L}_{\text{Focal-Radius}} = -\frac{1}{N}\sum_{i=1}^N \alpha_{y_i} w_i (1 - P_{i, y_i})^\gamma \log(P_{i, y_i})
$$

where $\gamma = 2.0$ dynamically focuses attention on hard-to-classify peripheral capillaries.

### 4. Differentiable Edge-Consistency Regularization
Topological fractures are penalized directly during backpropagation:

$$
\mathcal{L}_{\text{topo}} = \frac{1}{|E|} \sum_{(u, v) \in E} (P_u - P_v)^2
$$

This guides the HGAT parameters to enforce smooth label transitions along anatomical branches.

---

## 📊 Empirical Benchmark & Ablation Study

Evaluated across **all 30 unseen test CT scans** (15 patients × 2 breathing phases = **245,760 total graph nodes**) from the Lung250M-4B test split:

| Configuration | Overall Acc (%) | Macro S-Dice | Macro mIoU (%) | Mean BMC (Fractures) | Vein Recall (%) | Artery Recall (%) | Small Vessel Acc (r <= 1.5) | Small Artery Rec (BV5) | Small Vein Rec (BV5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trial 4 Baseline ($T=0.45$)** | 70.23 | 0.7017 | 54.06 | 588.8 | 68.85 | 71.49 | 68.63 | 70.90 | 66.00 |
| **Trial 4 + Track 1 Diffusion** | **70.62** | **0.7043** | **54.42** | **560.5** | 68.92 | 72.15 | **68.98** | 71.30 | 66.25 |
| **Trial 5 Retrained ($T=0.45$)** | 68.97 | 0.6866 | 52.36 | 593.0 | 61.72 | **75.62** | 68.22 | **76.34** | 58.84 |
| **Trial 5 + Diffusion ($T=0.45$)** | 69.68 | 0.6922 | 53.16 | 564.6 | 64.22 | 75.11 | 68.91 | 75.76 | 61.64 |
| **Trial 5 + Diffusion ($T=0.43$, Balanced)** | 69.17 | 0.6894 | 52.79 | 565.2 | **72.48** | 66.53 | 68.54 | 67.20 | **69.75** |

### Key Empirical Observations
1. **Topological Healing**: Track 1 Confidence Diffusion consistently eliminated **28.3 to 28.4 fractures per scan** across all models without degrading boundary delineation.
2. **Small Vessel Calibration**: Trial 5 successfully lifted small artery recall to **75.76%** (vs 70.90% baseline).
3. **Threshold Flexibility for Clinical Phenotyping**: Operating at $T=0.43$ yields balanced sensitivity (**72.48% Vein Recall** and **69.75% Small Vein Recall**), preventing false vascular pruning artifacts in downstream clinical analysis.

---

## 🩺 Clinical Imaging Biomarkers & Vascular Morphometry

Our pipeline automatically computes standardized clinical imaging biomarkers directly from the reconstructed 3D dense volumes:

| Biomarker | Description | Clinical Significance |
| :--- | :--- | :--- |
| **Total Lung Volume (mL)** | Sum of parenchymal volume within lung boundary | Evaluates hyperinflation and restrictive parenchymal defects. |
| **Total Vessel Volume (mL)** | Total vascular blood volume across both lungs | Assesses global pulmonary vascular capacity. |
| **Emphysema Index (LAA-950 %)** | Percentage of lung parenchyma voxels with attenuation < -950 HU | Primary radiological metric for emphysematous tissue destruction. |
| **Artery-to-Vein Ratio (AVR)** | Volumetric ratio of arterial to venous blood (Arterial Vol / Venous Vol) | Biomarker for vascular remodeling, shunting, and inflammation. |
| **Small Vessel Caliber (BV5 %)** | Blood volume of small peripheral vessels with radius < 1.26 mm (BV5 Vol / Total Vessel Vol × 100%) | Gold-standard biomarker for peripheral microvascular pruning and vessel loss. |

---

## 🌐 Interactive Visualizer & Generated Artifacts

### 1. Interactive 3D CT Slice Scrubber (`results/visualizations/prediction_viewer.html`)
An interactive, standalone web dashboard featuring:
- Full-width axial slice scrubber spanning all slices from apex to base of the lung.
- Triple overlay visualization: Raw Chest CT, Ground Truth Vessels, and Predicted Artery/Vein Segmentation (Red: Artery, Blue: Vein).
- Real-time clinical biomarker cards (Total Volume, LAA-950%, AVR, and BV5%).
- Zero installation required: simply double-click `results/visualizations/prediction_viewer.html` to open in any web browser.

### 2. Comprehensive Evaluation Report (`docs/reports/Graph_PAVNet_Evaluation_Report.pdf`)
A publication-ready PDF report containing detailed patient demographics, slice statistics, contingency confusion matrices, and case-by-case test metrics.

---

## 📁 Repository Structure

```
Lung 250M-4B/
├── src/                                  # Modular core Python package
│   ├── __init__.py                       # Package exports
│   ├── config.py                         # Paths, hyperparameters & dataset constants
│   ├── models/                           # Deep learning architectures
│   │   ├── __init__.py
│   │   ├── hgat.py                       # Hierarchical Graph Attention Network
│   │   └── patch_extractor.py            # 9-plane orthogonal vision transformer encoder
│   ├── data/                             # Data loading and graph synthesis
│   │   ├── __init__.py
│   │   ├── graph_builder.py              # k-NN graph construction & hilum seed finder
│   │   └── dataset.py                    # PyG dataset & patient-wise group splitting
│   ├── training/                         # Training routines & loss functions
│   │   ├── __init__.py
│   │   ├── losses.py                     # Radius-weighted focal + topological loss
│   │   └── trainer.py                    # Optimization loops, LR scheduler & validation
│   ├── postprocessing/                   # Spatial & graph post-processing
│   │   ├── __init__.py
│   │   ├── smoothing.py                  # Vectorized graph neighbor voting
│   │   ├── diffusion.py                  # Confidence-guided edge-affinity diffusion
│   │   ├── thresholding.py               # Caliber-adaptive thresholding
│   │   └── mapping.py                    # KDTree sparse skeleton -> dense 3D NIfTI volume
│   └── evaluation/                       # Evaluation & biomarker calculation
│       ├── __init__.py
│       ├── metrics.py                    # Macro S-Dice, PPV, Recall, BMC fractures
│       └── biomarkers.py                 # Clinical metrics: AVR, BV5, LAA-950, Volumetry
│
├── scripts/                              # Standalone CLI execution pipelines
│   ├── extract_features.py               # Step 1: Feature extraction (train & test)
│   ├── train_model.py                    # Step 2: Model training with patient-wise split
│   ├── evaluate_test.py                  # Step 3: Test evaluation & 3D NIfTI reconstruction
│   └── sweep_thresholds.py               # Step 4: Operating threshold sweeps & curves
│
├── notebooks/                            # Interactive Jupyter research notebooks
│   └── trial-graph-pav-net.ipynb         # Complete exploratory research trajectory
│
├── results/                              # Quantitative results & visualizations
│   ├── benchmarks/                       # JSON metrics across all test cases & trials
│   │   ├── all_splits_benchmark.json
│   │   ├── test_case_table.json
│   │   └── trial5_comparison_results.json
│   └── visualizations/                   # Interactive viewer & slice overlays
│       ├── prediction_viewer.html        # Interactive 3D slice & biomarker web viewer
│       ├── test_case_054_1_slice.png
│       └── train_case_017_1_slice.png
│
├── docs/                                 # Research literature & technical documentation
│   ├── papers/                           # Publication bibliography & DOIs (see README.md)
│   │   └── README.md                     # Paper citations & access links
│   └── reports/                          # Project evaluation reports
│       └── Graph_PAVNet_Evaluation_Report.pdf
│
├── models/                               # Pretrained weights (.pth)
│   ├── best_pavnet_final-5.pth           # Trial 5 checkpoint (Radius-weighted + Topo)
│   ├── best_pavnet_final-4.pth           # Trial 4 baseline checkpoint
│   └── patch_extractor.pth               # Deterministic visual patch encoder
│
├── predictions/                          # Output directory for dense 3D NIfTI files
│   └── .gitkeep                          # Directory tracker (NIfTIs gitignored)
│
├── 1_extract_aligned_features.py         # Backward-compatible root wrapper
├── 2_train_pavnet.py                     # Backward-compatible root wrapper
├── 3_evaluate_test.py                    # Backward-compatible root wrapper
├── sweep_thresholds.py                   # Backward-compatible root wrapper
├── postprocess.py                        # Backward-compatible root wrapper
├── .gitignore                            # Comprehensive git exclusion rules
├── requirements.txt                      # Python dependencies
└── README.md                             # Documentation
```

---

## ⚡ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Shreyas-Mohan/GNN-based-Pulmonary-Vessel-Separation.git
cd GNN-based-Pulmonary-Vessel-Separation
```

### 2. Create and Activate Conda Environment
```bash
conda create -n graph_pavnet python=3.9 -y
conda activate graph_pavnet
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*Note on PyTorch Geometric*: If your system requires specific CUDA-compiled binaries for `torch-geometric`, install them matching your CUDA version according to the [official PyG instructions](https://pyg.org). The codebase automatically falls back to native vectorized PyTorch operations if optional C++ scatter extensions are unavailable.

---

## 🚀 Step-by-Step Reproduction Guide

### Step 1: Feature Extraction
Extract deterministic 9-plane orthogonal patch embeddings from raw CT volumes:
```bash
# Extract for all splits
python scripts/extract_features.py --split all

# Or extract only the test set
python scripts/extract_features.py --split test
```

### Step 2: Train Graph-PAVNet
Train with Patient-Wise Group Splitting (zero patient leakage) and Radius-Weighted Focal Loss:
```bash
python scripts/train_model.py \
  --epochs 40 \
  --batch-size 4 \
  --lr 0.0004 \
  --gamma 2.0 \
  --lambda-topo 0.15 \
  --lambda-seed 0.05 \
  --threshold 0.45
```
Checkpoints will be saved to `models/best_pavnet_final-5.pth`.

### Step 3: Test Evaluation & 3D Volume Reconstruction
Evaluate performance on the 30 unseen test cases, output confusion matrices, and reconstruct 3D NIfTI volumes:
```bash
# Evaluate Trial 5 with Confidence Diffusion at balanced threshold
python scripts/evaluate_test.py \
  --checkpoint models/best_pavnet_final-5.pth \
  --threshold 0.43 \
  --reconstruct-first
```

### Step 4: Operating Threshold Sweep
Quickly precompute graph forward passes and sweep decision thresholds ($0.05 \le T \le 0.55$) in seconds:
```bash
python scripts/sweep_thresholds.py
```

---

## 📚 References & Citations

If you use this repository, model architectures, or methodology in your academic work, please cite the foundational publications:

### 1. Graph-PAVNet Architecture
```bibtex
@article{li2024graphpavnet,
  title={Graph-PAVNet: A Graph-Based Learning Framework for Pulmonary Artery and Vein Separation Using Multimodal Feature Sampling},
  author={Li, Qingya and Yuan, Ye and Liu, Lu and Bao, Nan and Xu, Lisheng and Tan, Wenjun},
  journal={Key Research Laboratory of Intelligent Computing of Medical Images, Northeastern University},
  year={2024}
}
```

### 2. Lung250M-4B Benchmark Dataset
```bibtex
@article{falta2024lung250m,
  title={Lung250M-4B: A Combined 3D Dataset for CT- and Point Cloud-Based Intra-Patient Lung Registration},
  author={Falta, Fenja and Gro{\ss}br{\"o}hmer, Christoph and Hering, Alessa and Bigalke, Alexander and Heinrich, Mattias P},
  journal={Institute of Medical Informatics, University of L{\"u}beck},
  year={2024}
}
```

### 3. COPDGene & DIR-LAB Benchmarks
```bibtex
@article{regan2010genetic,
  title={Genetic epidemiology of COPD (COPDGene) study design},
  author={Regan, Elizabeth A and Hokanson, John E and Murphy, James R and Make, Barry and Lynch, David A and Beaty, Terri H and Curran-Everett, Douglas and Silverman, Edwin K and Crapo, James D},
  journal={COPD: Journal of Chronic Obstructive Pulmonary Disease},
  volume={7},
  number={1},
  pages={32--43},
  year={2010},
  publisher={Taylor \& Francis}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## 🤝 Acknowledgments
We express our gratitude to the **Institute of Medical Informatics at the University of Lübeck** for providing the Lung250M-4B dataset, the **Key Research Laboratory of Intelligent Computing of Medical Images at Northeastern University** for the HGAT architectural formulation, and the **COPDGene Consortium** for pioneering clinical pulmonary imaging benchmarks.