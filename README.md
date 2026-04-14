"""# ⚖️ NodeBias

**An End-to-End AI Fairness Audit & Active Mitigation Microservice** *Built for the 2026 Google Solution Challenge*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-green.svg)](https://flask.palletsprojects.com/)
[![Gemini API](https://img.shields.io/badge/Google%20AI-Gemini%202.5%20Flash-orange.svg)](https://ai.google.dev/)

Machine learning models often inherit and amplify historical human biases. Existing fairness tools are opaque "black boxes" that only *detect* bias after the fact. **NodeBias** solves this by offering an API that not only audits models for Disparate Impact but actively mitigates that bias using mathematical reweighing before deployment.

---

## ✨ Core Features

* **🔍 Universal Telemetry Scanner:** Automatically detects protected demographic attributes (like gender, race, or age) and proxy variables using hybrid header/value heuristics.
* **🛠️ GlassBoxML (Custom Engine):** A proprietary machine learning framework built from absolute scratch in pure Python/NumPy. Features a custom B-Tree storage engine, a bespoke Random Forest implementation, and a custom Momentum Optimizer for Logistic Regression.
* **🔀 Hybrid ML Router:** Dynamically routes baseline tests through our custom GlassBoxML framework, and routes active mitigation tests through Scikit-Learn to guarantee mathematical constraint processing.
* **⚖️ Algorithmic Reweighing:** Balances historical dataset bias by mathematically adjusting the weights of disadvantaged groups to clear the 80% EEOC fairness threshold.
* **🤖 AI Fairness Consultant:** Integrates the **Google Gemini 2.5 Flash API** to automatically generate human-readable, plain-English executive summaries of mathematical audit results.

---

## 🏗️ Architecture

1.  **Data Upload:** Send a CSV via the `/api/audit` endpoint.
2.  **Telemetry:** The scanner checks for sensitive proxy columns.
3.  **Baseline Run:** Data is routed to the custom `GlassBoxML` engine to measure the raw Disparate Impact Ratio (DIR).
4.  **Mitigation:** If bias is detected, the `mitigation.py` engine calculates balancing weights based on mathematical group probability.
5.  **Re-Train:** Data is routed to Scikit-Learn models that apply the calculated weights during training.
6.  **AI Report:** The final metrics are sent to Google Gemini, which returns an executive summary for the dashboard.

---

## 🚀 Quick Start (Local Development)

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/NodeBias.git](https://github.com/your-username/NodeBias.git)
cd NodeBias
