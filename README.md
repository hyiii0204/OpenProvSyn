# ProvSyn

This project provides a provenance graph synthesis framework including **graph structure synthesis**, **textual attribute synthesis**, and **multi-dimensional fidelity evaluation**.

Before running the code, please make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## 1. Graph Structure Synthesis

```bash
cd structure
python main.py                 # Train the graph generation model
python evaluate.py             # Generate graphs and compute MMD, Novelty, and Uniqueness
```

---

## 2. Text Attribute Synthesis

```bash
cd text
python lora_data.py            # Prepare training data
# Refer to training.ipynb for model training
# Refer to inference.ipynb for attribute synthesis
```

## 3. Fidelity Evaluation

```bash
cd fidelity
```

| Dimension          | Script                                    |
| ------------------ | ----------------------------------------- |
| **Text Attribute** | `python text.py`                          |
| **Temporal**       | `python lcs.py`, `python dtw.py`          |
| **Embedding**      | `python netlsd.py`, `python graph2vec.py` |
| **Semantic**       | `python semantic.py`                      |

