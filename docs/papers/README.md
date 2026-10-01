# Reference Literature & Publications

This directory documents the foundational publications and benchmark datasets underlying this project. To respect publisher copyright, raw PDF files are excluded from Git tracking via `.gitignore`. You can access each publication via its respective DOI or official repository link below:

---

### 1. Graph-PAVNet (Core Model Architecture)
- **Title**: *Graph-PAVNet: A Graph-Based Learning Framework for Pulmonary Artery and Vein Separation Using Multimodal Feature Sampling*
- **Authors**: Qingya Li, Ye Yuan, Lu Liu, Nan Bao, Lisheng Xu, Wenjun Tan
- **Affiliation**: Key Research Laboratory of Intelligent Computing of Medical Images, Ministry of Education, Northeastern University, China.
- **Summary**: Introduces Light Vessel Structured Modelling (LVSM) and Hierarchical Graph Attention Networks (HGAT) with BFS layer propagation for topological pulmonary artery-vein separation.

---

### 2. Lung250M-4B (Dataset Benchmark)
- **Title**: *Lung250M-4B: A Combined 3D Dataset for CT- and Point Cloud-Based Intra-Patient Lung Registration*
- **Authors**: Fenja Falta, Christoph Großbröhmer, Alessa Hering, Alexander Bigalke, Mattias P. Heinrich
- **Affiliation**: Institute of Medical Informatics, University of Lübeck & Radboud University Medical Center, Nijmegen.
- **Benchmark Source**: Built upon the DIR-LAB COPDgene benchmark (in- and expiratory breath-hold CT pairs) and PVT1010 vascular point cloud geometry.
- **Dataset Access**: [https://grand-challenge.org](https://grand-challenge.org) / [University of Lübeck Medical Informatics](https://www.imi.uni-luebeck.de)

---

### 3. Related GNN & Volumetric Segmentation Literature
- **Nature Scientific Reports (2025)**: *Graph neural network model using radiomics for lung CT image segmentation (GEANet)*
  - Mohammad Khalid Faizi, Yan Qiang, et al.
  - [DOI: 10.1038/s41598-025-12141-0](https://doi.org/10.1038/s41598-025-12141-0)
- **Transformer-based 3D U-Net**:
  - Multi-planar volumetric segmentation baseline for thoracic CT imaging and pulmonary vessel segmentation.
