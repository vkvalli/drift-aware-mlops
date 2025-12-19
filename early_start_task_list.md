# Early-Start Task Plan for Thesis Project

This markdown file outlines all the tasks you should begin **now**, *before* the official semester project schedule starts. Each section includes what to do, how to do it, and recommended free/open‑source tools suitable for students.

---

## ✅ 1. Environment Setup (Critical First Step)
Setting up your environment early prevents delays once development begins. Use tools that are **open-source** and free for students.

### **1.1 Programming Languages & Package Managers**
- **Python 3.10+** – Primary development language
- **pip / conda** – Package & environment management
  - Recommended: **Miniconda** (free, lightweight)
 
### **1.2 Machine Learning Libraries** (All open-source)
- **PyTorch** – Preferred for RL + deep learning
- **scikit-learn** – Baselines + preprocessing
- **TensorFlow** (optional) – If you want comparison frameworks

### **1.3 Tracking, Logging & Orchestration Tools**
| Component | Recommended Tool | Free? | Reason |
|----------|------------------|-------|--------|
| Experiment Tracking | **MLflow** | ✔ | Open-source, perfect for versioning models/metrics |
| Workflow Orchestration | **Prefect** or **LangGraph** | ✔ | Prefect Cloud free tier is sufficient; LangGraph is fully free |
| Containerization | **Docker** | ✔ | Industry standard |
| Observability | **Prometheus + Grafana** | ✔ | Metrics + dashboarding |
| Energy Tracking | **CodeCarbon** | ✔ | Carbon/kWh tracking |

### **1.4 GPU / Cloud Compute Options**
| Provider | Free Option | Notes |
|----------|-------------|-------|
| **Google Colab** | Free tier w/ some GPU access | Good for experimentation |
| **Google Cloud Platform** | $300 credits for students | You can run small VM instances |
| **Paperspace Gradient** | Free GPU instances (limited) | Good for prototyping |
| **Local GPU (if available)** | Free | Most convenient |

### **1.5 Setup Tasks**
- Install Python + Conda
- Create a new environment: `conda create -n mlops-agent python=3.10`
- Install MLflow, PyTorch, CodeCarbon, scikit-learn
- Install Docker Desktop
- Install Prometheus & Grafana via Docker Compose
- Create a folder structure:
```
project_root/
│── data/
│── notebooks/
│── src/
│   ├── ingestion/
│   ├── drift/
│   ├── rl_agent/
│   ├── utils/
│── models/
│── experiments/
│── dashboards/
│── docker/
│── README.md
```

---

## ✅ 2. Build the Ingestion + Logging Skeleton
This sets your foundation for all future components.

### **Tasks:**
- Write a basic script in `src/ingestion/` that:
  - Accepts data batches
  - Cleans/normalizes simple tabular input
  - Logs the batch to MLflow (dummy run)
- Set up Prometheus metric endpoints:
  - `ingestion_batch_count`
  - `ingestion_latency`
- Connect Grafana dashboard to Prometheus

### **Goal:**
A minimal pipeline that *runs from end to end* even if it uses placeholder logic.

---

## ✅ 3. Implement a Dummy Drift Detection Pipeline
Before building real drift detection, you need the interface + flow.

### **Tasks:**
- Create a drift detection module with functions:
  - `compute_drift_score(batch)` → returns a random float
  - `log_drift_score_to_mlflow(score)`
- Emit Prometheus metrics:
  - `drift_score`
  - `drift_alerts_triggered`
- Add drift score graphs to Grafana

### **Goal:**
Your system behaves *as if* drift exists → lets you test RL integration later.

---

## ✅ 4. Prepare Benchmark Datasets
Download and organize all datasets you plan to use.

### **Datasets to prepare:**
#### Tabular:
- UCI Electricity
- Airline Delays
- Synthetic financial drift dataset

#### Text:
- AG News
- IMDB drift variations

#### Image:
- CIFAR-10C
- Rotated-MNIST

### **Tasks:**
- Create a loader script per dataset
- Preprocess into consistent batch formats
- Split into train/val/test
- Store metadata as JSON (schema, classes, etc.)

### **Goal:**
All datasets ready for experimentation *before coding begins*.

---

## ✅ 5. Draft the RL Reward Formulation
You don’t need to implement RL yet — just define the math & structure.

### **Define the following:**
- **State vector:**
  - drift score
  - cost estimate
  - energy estimate
  - fairness deviation
  - model accuracy
- **Actions:** `{retrain, fine_tune, defer}`
- **Reward function:**
  - penalize high cost/CO₂
  - penalize performance drop
  - penalize high fairness deviation
  - reward stable accuracy

### **Goal:**
A complete conceptual specification for your agent.

---

## ✅ 6. Build the Evaluation Scaffold
Even with no real values yet, implement function signatures such as:
```
def evaluate_accuracy(model, data): ...
def evaluate_fairness(metrics): ...
def evaluate_energy_usage(job_id): ...
def compute_drift_at_risk(drift_scores): ...
```
This ensures the pipeline remains modular and plug‑and‑play.

---

## ✅ 7. Set Up Progress Tracking & Documentation
This will be required all semester.

### **Tasks:**
- Create a **biweekly progress report template** with:
  - Tasks completed
  - Issues encountered
  - Adjustments to schedule
- Set up a **GitHub Project board** (To‑Do / Doing / Done)
- Create documentation folders:
  - `/docs/research_log.md`
  - `/docs/design_decisions.md`
  - `/docs/meeting_notes.md`

### **Goal:**
Smooth communication with your advisor and clean documentation for thesis writing.

---

## ✅ 8. Improve Version Control & Repo Structure
### **Tasks:**
- Create a private GitHub repo
- Use branch structure:
  - `main`
  - `dev`
  - feature branches: `feat/drift-monitor`, `feat/ingestion`, etc.
- Add CI linters:
  - `flake8`
  - `black`
  - `pytest` (later for unit tests)

---

## OPTIONAL: Expand Literature Review
If you want a head start on the thesis writing:
- Annotate 5–8 more 2024–2025 drift/MLOps papers
- Write small summaries for each
- Add them to `/docs/literature_notes.md`

---

# 🎯 Summary of Early‑Start Deliverables
- Complete environment setup with open-source tools
- Skeleton pipeline for ingestion + logging + drift
- Dataset download + initial preprocessing
- Draft RL reward formulation
- Evaluation metric stubs
- Organized repo + documentation system
- Monitoring stack (Prometheus/Grafana) running in Docker

---

If you want, I can also generate:
- The **folder structure** in code
- Initial `.py` files for each component
- A **complete README.md** for the GitHub repo
- A progress report template
- A docker-compose file for MLflow + Prometheus + Grafana

