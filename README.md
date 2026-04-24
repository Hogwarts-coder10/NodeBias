# ⚖️ NodeBias

> 🏆 Built to ensure fair, transparent, and accountable AI systems.

**An End-to-End AI Fairness Audit & Active Mitigation Microservice**
*An End-to-End Clinical Fairness Audit & Active Mitigation Microservice Built for the 2026 Google Solution Challenge*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-green.svg)](https://flask.palletsprojects.com/)
[![Gemini API](https://img.shields.io/badge/Google%20AI-Gemini%202.0%20Flash-orange.svg)](https://ai.google.dev/)


## 🚀 Deployment & Tech
Backend: Flask / Gunicorn (Render) optimized for 512MB RAM.

Frontend: React / Vite (Vercel) for rapid clinical reporting.

AI Engine: Hybrid Router (GlassBoxML + Scikit-Learn) + Gemini 2.0 Flash.

# ⚖️ NodeBias
[**Live Demo 🚀**](https://nodebias.vercel.app/)

---

## 🌍 Problem Statement

Modern AI systems influence high-stakes decisions such as hiring, loan approvals, and healthcare. However, these systems often inherit and amplify historical biases present in training data, leading to unfair and discriminatory outcomes.

Ensuring fairness in AI is not just a technical challenge—it is a societal necessity.

---

## 🏥 Medical Context & Problem Statement
In healthcare, AI models are used for readmission prediction, triage, and diagnosis. However, if these models are biased against specific genders, ethnicities, or age groups, they can lead to disparate health outcomes. NodeBias was engineered to provide medical professionals with a "GlassBox" view of their models, ensuring that clinical decisions are based on data, not systemic bias.

## 🇺🇳 UN SDG Alignment
NodeBias directly supports Goal 3: Good Health and Well-being and Goal 10: Reduced Inequalities. By auditing clinical models, we ensure that technological advancement in medicine does not come at the cost of equity.

## 💡 Solution: NodeBias
NodeBias is a specialized fairness microservice for medical AI that:

🔍 Clinical Telemetry: Detects bias in medical datasets (e.g., patient demographics).

⚖️ Safety Audits: Quantifies fairness using the Disparate Impact Ratio (DIR).

🛠️ Active Mitigation: Uses Algorithmic Reweighing to correct bias before deployment.

🤖 Clinical Summaries: Explains audit results in plain English for hospital administrators and doctors.

---


## ✨ Core Features
🛠️ GlassBoxML (Custom Engine)
A custom-built ML framework designed for Auditability. Unlike "Black Box" models, GlassBoxML allows clinicians to trace how demographic features influence predictions, ensuring a transparent baseline before any mitigation is applied.

⚖️ Medical Grade Mitigation
Utilizes Algorithmic Reweighing to balance parity. In our tests on the Diabetes Readmission Dataset, we successfully moved a biased model to a 0.993 DIR, achieving near-perfect mathematical equity across gender lines.

🤖 Gemini-Powered Reporting
Leverages Google Gemini 2.0 Flash to bridge the gap between data science and clinical practice. It generates professional audit summaries that explain why a model is safe for deployment.

### 🔀 Hybrid ML Router

Routes tasks intelligently:

* GlassBoxML → transparency & baseline analysis
* Scikit-Learn → reliable mitigation and retraining

---

### ⚖️ Algorithmic Reweighing

Applies mathematical reweighting to ensure fairness:

* Balances underrepresented groups
* Targets the **80% Disparate Impact threshold**

---

### 🤖 Explainability Layer (Gemini API)

Uses **Google Gemini 2.5 Flash** to:

* Translate metrics into plain English
* Generate executive summaries
* Bridge technical → human understanding

---


## 🏗️ System Architecture

```
Data → Audit → Detect Bias → Mitigate → Re-evaluate → Explain
```

### Workflow:

1. Upload dataset via `/api/audit`
2. Detect sensitive attributes
3. Run baseline model (GlassBoxML)
4. Compute fairness metrics (Disparate Impact)
5. Apply mitigation (reweighing)
6. Retrain model (Scikit-Learn)
7. Generate explanation using Gemini API

---

## 📊 Example Output

**Before Mitigation:**

* Disparate Impact Ratio: 0.62 ⚠ (Biased)

**After Mitigation:**

* Disparate Impact Ratio: 0.91 ✅ (Fair)

---

## 🧠 Tech Stack

### Backend

* Python
* Flask
* NumPy / Pandas
* Scikit-Learn

### AI / ML

* GlassBoxML (Custom Framework)
* Google Gemini API

### Frontend

* ViteJS

### Deployment

* Google Cloud / Firebase

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Hogwarts-coder10/NodeBias.git
cd NodeBias
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the Server

```bash
python app.py
```

---

### 4. Test API

Send a CSV file to:

```
POST /api/audit
```

---

## 📈 Key Metrics Used

* **Statistical Parity**
* **Disparate Impact Ratio (DIR)**
* **Group-wise Prediction Rates**

---

## 🏆 Why NodeBias Stands Out

* ✔ Combines **custom ML + industry tools**
* ✔ Goes beyond detection → **active mitigation**
* ✔ Provides **explainable AI outputs**
* ✔ Designed for **real-world deployment**

---

## 🔮 Future Improvements

* Real-time bias monitoring
* Support for more fairness metrics
* Model-agnostic explainability (SHAP/LIME)
* Full SaaS deployment

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork and improve.

---

## 📜 License

MIT License

---

## 🙌 Acknowledgements

* Google Solution Challenge
* Open-source ML community
* Research on AI fairness and ethics

---

## 💬 Final Note

> Fair AI is not optional—it is essential.
> NodeBias ensures that intelligent systems remain just, transparent, and accountable.
