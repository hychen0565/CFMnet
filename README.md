# CFMANet
### Structural Affinity Orchestrates Transfer: Few-Shot Defect Segmentation via Morphology-Adaptive Evidence Construction and Correspondence-Regulated Transfer

> **Submitted to IEEE Transactions on Industrial Informatics (TII) | Under Review**
>
> Official PyTorch implementation for cross-granularity few-shot defect segmentation on heterogeneous metallic surfaces.

---

## 📌 Introduction
Pixel-level defect segmentation under limited annotation is a core bottleneck for intelligent industrial quality inspection. Existing few-shot methods face two critical challenges in real production scenarios:
1. The semantic granularity gap between coarsely-labeled production archives and fine-grained inspection targets
2. Poor transfer robustness caused by material appearance variations and diverse defect morphologies

This work proposes **CFMANet (Cross-Granularity Frequency-Aware Matching and Alignment Network)**, which couples morphology-adaptive evidence construction with compatibility-guided knowledge transfer. It achieves accurate few-shot defect segmentation across different semantic granularities and heterogeneous non-ferrous metal surfaces.

---

## ✨ Core Highlights
- **Morphology-Adaptive Spectral Routing (MASR)**: Hierarchically selects directional and frequency-band evidence, and aggregates non-local spatial relations to construct structurally-preserved support prototypes.
- **Entropic Correspondence Modulation Decoder (ECMD)**: Estimates soft patch-wise correspondences via Sinkhorn-style normalization, and adaptively regulates prototype guidance strength throughout query decoding.
- 
- **NFM9 Benchmark**: A 9-class non-ferrous metal defect dataset with pixel-level annotations, collected from real-world production lines covering copper, brass and aluminum alloys.
- **State-of-the-art Performance**: Achieves leading segmentation accuracy on CGFSDS-9, NFM9 and FSSD-12 benchmarks under both 1-shot and 5-shot settings.

---

## 🧠 Method Overview
CFMANet follows a shared support-query backbone architecture, with two core complementary modules:

1. **MASR Module**  
Decomposes support features via Haar wavelet transform, performs support-adaptive directional and frequency-band routing, and organizes spectral evidence into spatially structured prototypes via non-local relational aggregation.

2. **ECMD Module**  
Establishes soft patch correspondences between support prototypes and query features. It modulates support guidance strength according to patch compatibility, realizing spatially-selective knowledge injection during hierarchical decoding.
---

## 📊 Datasets
We evaluate our method on three benchmarks covering both cross-granularity and same-granularity few-shot segmentation tasks:

| Dataset | Task Scenario | Defect Categories | Description |
|---------|---------------|-------------------|-------------|
| CGFSDS-9 | Cross-granularity transfer | 6 fine-grained defects | Source: strip steel, aluminum profiles, magnetic tiles<br>Target: seamless steel tube inner surfaces |
| FSSD-12 | Same-granularity transfer | 12 strip steel defects | Standard benchmark for industrial few-shot defect segmentation |
| **NFM9** | Cross-material transfer | 9 non-ferrous metal defects | Proposed in this work, collected from real production lines with polarized line-scan imaging |

> The full NFM9 dataset and training code will be made publicly available upon paper acceptance.

---

## ⚙️ Environment Requirements
- Python 3.8+
- PyTorch 1.10+ / torchvision 0.11+
- CUDA 11.0+
- NumPy, OpenCV-Python, SciPy, tqdm

### Installation
```bash
git clone https://github.com/your-username/CFMANet.git
cd CFMANet
pip install -r requirements.txt
```

---

## 🚀 Quick Start
### 1. Data Preparation
Organize the dataset directory as follows:
```
data/
├── CGFSDS9/
│   ├── support/
│   ├── query/
│   └── masks/
├── NFM9/
└── FSSD12/
```

### 2. Training
Train on CGFSDS-9 under 1-shot setting:
```bash
python train.py --dataset CGFSDS9 --n_shot 1 --batch_size 4 --epochs 100 --lr 1e-4
```

Train on NFM9 under 5-shot setting:
```bash
python train.py --dataset NFM9 --n_shot 5 --batch_size 4 --epochs 100 --lr 1e-4
```

### 3. Evaluation
```bash
python test.py --dataset CGFSDS9 --n_shot 1 --checkpoint ./checkpoints/cfmanet_cgfsds9_1shot.pth
```

---

## 📈 Quantitative Results
Performance comparison with state-of-the-art methods (mIoU, %):

### CGFSDS-9 (Cross-Granularity)
| Method | 1-shot | 5-shot |
|--------|--------|--------|
| LGPR (TII 2026) | 74.43 | 75.09 |
| PANDA (TII 2026) | 73.86 | 74.57 |
| **CFMANet (Ours)** | **76.15** | **77.13** |

### NFM9 (Heterogeneous Metallic Surfaces)
| Method | 1-shot | 5-shot |
|--------|--------|--------|
| PANDA (TII 2026) | 71.42 | 72.34 |
| RDPrompter (TII 2026) | 71.21 | 72.46 |
| **CFMANet (Ours)** | **73.94** | **74.87** |

### FSSD-12
| Method | 1-shot | 5-shot |
|--------|--------|--------|
| LGPR (TIP 2026) | 66.92 | 68.90 |
| MAPTNet (TIM 2025) | 66.47 | 68.00 |
| **CFMANet (Ours)** | **68.34** | **70.14** |

---

## 📝 Citation
If this work is helpful for your research, please consider citing our paper:

```bibtex
@article{cfmanet2026,
  title={Structural Affinity Orchestrates Transfer: Few-Shot Defect Segmentation via Morphology-Adaptive Evidence Construction and Correspondence-Regulated Transfer},
  author={Your Name and Co-authors},
  journal={IEEE Transactions on Industrial Informatics},
  year={2026},
  note={Under Review}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact
If you have any questions, please feel free to open an issue or contact us via email.
