[Vietnamese](README.md) | English

<h1 id="english" align="center">GNN–LLMs & X-Hacking Demo</h1>
<h3 align="center">Exploring GNNs, LLMs and X-Hacking Risks in AutoML</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue"/>
  <img src="https://img.shields.io/badge/Explainability-SHAP-green"/>
  <img src="https://img.shields.io/badge/ML-RandomForest%20%7C%20LogReg-orange"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen"/>
</p>

<br>

<div id="toc-en">

## 📌 Table of Contents

- [✨ Overview](#en-overview)
- [🧠 GNN–LLMs Interaction](#en-gnnllm)
- [⚠️ What is X-Hacking?](#en-xhacking)
- [🧪 Demo: Rashomon Effect](#en-demo)
- [📊 SHAP Comparison](#en-shap)
- [🚀 Run](#en-run)

</div>

---

<h2 id="en-overview">✨ Overview</h2>

This repository demonstrates:

- How **Large Language Models (LLMs)** support **Graph Neural Networks (GNNs)**  
  → Improving feature representation, enabling auto-labeling, and boosting generalization  
- How **GNNs support LLMs**  
  → Reducing hallucination through Knowledge Graphs, improving explainability  
- The risk of **X-Hacking** — intentionally selecting models with SHAP explanations that match a desired conclusion  
- A practical demo showing **two ML models with similar accuracy but drastically different explanations**

The project includes Python code and SHAP visualizations to clearly illustrate this phenomenon.

---

<h2 id="en-gnnllm">🧠 GNN–LLMs Interaction</h2>

### How LLMs Improve GNNs  
- ✨ Enhance text-based feature embeddings with rich semantics  
- 🏷️ Provide data labeling and few-shot augmentation  
- 🧩 Improve robustness when handling heterophily or out-of-distribution (OOD) data  

### How GNNs Support LLMs  
- 🔍 Reduce hallucinations through Knowledge Graph grounding  
- 📖 Provide interpretable reasoning paths via graph structures  

---

<h2 id="en-xhacking">⚠️ What is X-Hacking?</h2>

**X-Hacking** refers to:

> *Intentionally selecting a machine learning model whose SHAP explanation supports a predefined narrative, even when other models achieve the same accuracy.*

This occurs due to the **Rashomon Effect** - Multiple models may achieve **equivalent accuracy** while using **completely different internal logic**.

---

<h2 id="en-demo">🧪 Demo: Rashomon Effect</h2>

Using the **Breast Cancer Wisconsin dataset** (Scikit-Learn), we train two models:

- **Random Forest**  
- **Logistic Regression**

Both achieve approximately **96% accuracy**, but their SHAP-based feature importance rankings differ significantly → This discrepancy forms the foundation for X-Hacking.

---

<h2 id="en-shap">📊 SHAP Comparison</h2>

<p align="center">
  <img src="/images/SHAP_RF.png" width="45%">
  <img src="/images/SHAP_LR.png" width="45%">
</p>

### Quick Observations

- **Random Forest (RF)** identifies **worst concave points** as a top feature  
- **Logistic Regression (LR)** highlights **mean perimeter**, **area**, and **radius error**  
- Both models are “correct,” yet their explanations differ dramatically  

**→ A clear illustration of how X-Hacking becomes possible.**

---

<h2 id="en-run">🚀 Run</h2>

### 1️⃣ Install required libraries
```bash
pip install shap scikit-learn pandas matplotlib
```

### 2️⃣ Run the demo
```bash
python demo_xhacking.py
```
