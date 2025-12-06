<h1 align="center">GNN–LLMs & X-Hacking Demo</h1>
<h3 align="center">Tìm hiểu GNN, LLMs và rủi ro X-Hacking trong AutoML</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue"/>
  <img src="https://img.shields.io/badge/Explainability-SHAP-green"/>
  <img src="https://img.shields.io/badge/ML-RandomForest%20%7C%20LogReg-orange"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen"/>
</p>

<br>

<div id="toc">

## 📌 Mục lục

- [✨ Giới thiệu](#gioithieu)
- [🧠 Tương tác GNNs × LLMs](#gnnllm)
- [⚠️ X-Hacking là gì?](#xhacking)
- [🧪 Demo: Hiệu ứng Rashomon](#demo)
- [📊 So sánh SHAP](#shap)
- [🚀 Run](#run)

</div>

---

<h2 id="gioithieu">✨ Giới thiệu</h2>

Repository này minh họa:

- Cách **LLMs hỗ trợ GNNs**: cải thiện đặc trưng, sinh nhãn, tăng khả năng tổng quát hóa  
- Cách **GNNs hỗ trợ LLMs**: giảm hallucination bằng Knowledge Graph, tăng explainability  
- Rủi ro **X-Hacking** – chọn mô hình có lời giải thích (SHAP) phù hợp với kết luận mong muốn  
- Demo thực nghiệm với **2 mô hình accuracy tương đương nhưng giải thích khác nhau**

Repo cung cấp mã Python + biểu đồ SHAP để quan sát rõ ràng hiện tượng này.

---

<h2 id="gnnllm">🧠 Tương tác GNNs × LLMs</h2>

### LLMs ➜ Cải thiện GNNs  
- ✨ Nâng cao chất lượng embedding đặc trưng (text → rich semantic vectors)  
- 🏷️ Gán nhãn + tăng cường dữ liệu (few-shot)  
- 🧩 Giảm lỗi khi gặp dữ liệu phức tạp hoặc khác phân phối (OOD)  

### GNNs ➜ Tăng độ chính xác cho LLMs  
- 🔍 Giảm hallucination bằng Knowledge Graph  
- 📖 Minh bạch hơn với reasoning paths qua đồ thị  

---

<h2 id="xhacking">⚠️ X-Hacking là gì?</h2>

**X-Hacking** là hành vi:

> *Cố tình chọn mô hình có lời giải thích (SHAP) phù hợp với kết luận định sẵn, dù accuracy giữa các mô hình là như nhau.*

Điều này xảy ra do **hiệu ứng Rashomon** - Nhiều mô hình có **accuracy tương đương** nhưng **logic nội tại rất khác nhau**.

---

<h2 id="demo">🧪 Demo: Hiệu ứng Rashomon trong thực tế</h2>

Dùng bộ dữ liệu **Breast Cancer** (Scikit-learn), huấn luyện 2 mô hình:

- **Random Forest**  
- **Logistic Regression**

Cả hai đều đạt ~**96% accuracy**, nhưng SHAP Feature Importance lại *không hề giống nhau* → Chìa khoá cho X-Hacking.

---

<h2 id="shap">📊 So sánh SHAP</h2>

<p align="center">
  <img src="/images/SHAP_RF.png" width="45%">
  <img src="/images/SHAP_LR.png" width="45%">
</p>

### Nhận xét nhanh:

- Random Forest (RF) đánh giá **worst concave points** cực kỳ quan trọng  
- Logistic Regression (LR) lại coi trọng **mean perimeter**, **area**, **radius error**  
- Cả hai đều "đúng", nhưng chúng *giải thích theo hai cách khác nhau*  

**→ Bằng chứng trực quan về X-Hacking.**

---

<h2 id="run">🚀 Run</h2>

### 1️⃣ Cài đặt thư viện
```bash
pip install shap scikit-learn pandas matplotlib
```

### 2️⃣ Chạy file
```bash
python demo_xhacking.py
```
