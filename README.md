# MLOps Mini-Project: CallCenterAI - Intelligent Customer Ticket Classification

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)](https://www.docker.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800)](https://grafana.com/)

## 📋 About This Project

**CallCenterAI** is a complete MLOps solution that automatically classifies customer tickets from a call center (emails, chat, phone) into different business categories (e.g., *Billing*, *Technical Issue*, *Account Access*, etc.) using **two complementary NLP approaches**.

**🎯 Main Objective:**

Implement a complete MLOps solution to automatically classify customer tickets into different business categories using two NLP approaches:
- **Classic Model:** TF-IDF + SVM
- **Advanced Model:** Transformer (Hugging Face)

The system must be dockerized, monitored, integrated with a CI/CD pipeline, and include an AI agent to orchestrate predictions.

**🔑 Key Features:**

🤖 **Dual NLP Approach:**
- **Classic Model:** TF-IDF + SVM (scikit-learn) - Fast and efficient for standard cases
- **Advanced Model:** Multilingual Transformer (Hugging Face distilbert-base-multilingual-cased) - Powerful for complex and multilingual texts (supports 104 languages including FR and AR)

🧠 **Intelligent AI Agent:**
- Automatic request analysis (language, length, complexity)
- Smart routing to the most appropriate model (TF-IDF/SVM or Transformer)
- Confidence-based decision making with explanation

📊 **Complete MLOps Pipeline:**
- **DVC:** Data and pipeline version control
- **MLflow:** Experiment tracking, metrics, hyperparameters, and model registry (Production/Staging)
- **Docker & Docker Compose:** Full containerization of all services
- **CI/CD:** GitHub Actions (tests, linting, build, push images, security scanning)
- **Monitoring:** Prometheus + Grafana for real-time metrics and alerts

🚀 **Production-Ready Architecture:**
- FastAPI services for each model + AI agent (microservices architecture)
- Prometheus metrics endpoints integrated
- Unit and integration tests
- Clear README with deployment instructions
- Comprehensive Grafana dashboards (latency, requests, errors)

**Tech Stack:**
- **Language:** Python 3.11
- **API Framework:** FastAPI
- **NLP Models:** TF-IDF + SVM (scikit-learn), Transformer (Hugging Face transformers)
- **MLOps:** MLflow (tracking + registry), DVC (data versioning)
- **Containerization:** Docker, Docker Compose
- **CI/CD:** GitHub Actions (tests, linting, build, push images)
- **Monitoring:** Prometheus + Grafana
- **Quality & Security:** pytest, black, flake8, isort, Trivy (security scanning), Bandit

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Deliverables](#deliverables)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
  - [Clone Repository](#clone-repository)
  - [Environment Setup](#environment-setup)
  - [Docker Setup](#docker-setup)
- [Project Phases](#project-phases)
  - [Phase 1: Dataset Preparation](#phase-1-dataset-preparation)
  - [Phase 2: Model Training](#phase-2-model-training)
  - [Phase 3: API Services](#phase-3-api-services)
  - [Phase 4: Containerization](#phase-4-containerization)
  - [Phase 5: MLOps Pipeline](#phase-5-mlops-pipeline)
  - [Phase 6: Tests & Validation](#phase-6-tests--validation)
- [Project Components](#project-components)
  - [TF-IDF + SVM Model](#tf-idf--svm-model)
  - [Transformer Model](#transformer-model)
  - [Intelligent AI Agent](#intelligent-ai-agent)
  - [MLflow Tracking](#mlflow-tracking)
  - [Monitoring Stack](#monitoring-stack)
- [Running the Application](#running-the-application)
  - [Local Development](#local-development)
  - [Docker Deployment](#docker-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [API Documentation](#api-documentation)
  - [API Endpoints](#api-endpoints)
  - [Request/Response Examples](#requestresponse-examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Project Overview

CallCenterAI is a complete MLOps project that demonstrates industry best practices for deploying and maintaining machine learning models in production. The system classifies customer support tickets into business categories using two different NLP approaches, with an intelligent agent that routes requests to the most appropriate model.

**Business Problem:**

Call centers receive thousands of tickets daily through various channels (email, chat, phone). Manual classification is time-consuming, error-prone, and doesn't scale. CallCenterAI automates this process, ensuring tickets are routed to the right department quickly and accurately.

**Solution:**

A dual-model approach ensures optimal performance:
- **Simple, fast requests** → TF-IDF + SVM (low latency, high throughput)
- **Complex, multilingual requests** → Transformer model (high accuracy, context-aware)

The intelligent agent analyzes each request and routes it to the best model based on:
- Text length and complexity
- Language detection (multilingual support: FR, EN, AR, etc.)
- Confidence scores
- PII detection and scrubbing

> **Screenshot Placeholder:** Add project dashboard overview showing all components

---

## Architecture

The project follows a microservices architecture with the following components:

```
┌─────────────────────────────────────────┐
│         API Gateway / Load Balancer      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │   AI Agent      │
         │  (FastAPI)      │
         │  - Routing      │
         │  - PII Scrubbing│
         │  - Confidence   │
         └────────┬────────┘
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
┌──────────────┐      ┌──────────────┐
│  TF-IDF+SVM  │      │ Transformer  │
│   Service    │      │   Service    │
│  (FastAPI)   │      │  (FastAPI)   │
└──────┬───────┘      └──────┬───────┘
       │                     │
       └──────────┬──────────┘
                  ▼
         ┌─────────────────┐
         │  MLflow Server  │
         │  - Tracking     │
         │  - Registry     │
         └────────┬────────┘
                  │
         ┌────────┴────────┐
         ▼                 ▼
┌──────────────┐    ┌──────────────┐
│  Prometheus  │    │     DVC      │
│  (Metrics)   │    │   (Data)     │
└──────┬───────┘    └──────────────┘
       │
       ▼
┌──────────────┐
│   Grafana    │
│ (Dashboards) │
└──────────────┘
```
**Repository Structure**

![Alt text](images/img1.png)

**Component Communication:**
1. User request arrives at AI Agent
2. Agent analyzes request (language, length, complexity, PII)
3. Agent routes to appropriate model (TF-IDF/SVM or Transformer)
4. Model returns prediction with confidence score
5. Agent returns final prediction with explanation
6. All interactions logged to MLflow
7. Metrics exposed to Prometheus
8. Grafana visualizes metrics in real-time

---

## Deliverables

### ✅ Core Deliverables

**1. Structured GitHub Repository:**
- Training code (TF-IDF + SVM, Transformer fine-tuning)
- API services (FastAPI) for each model + AI agent
- Dockerfiles + docker-compose.yml
- MLOps pipeline (DVC, MLflow)
- CI/CD (GitHub Actions)
- Monitoring (Prometheus + Grafana)
- Unit and integration tests

**2. Clear README:**
- Setup and launch instructions
- API documentation
- Architecture explanation
- Usage examples

**3. MLflow Report:**
- Tracked experiments with metrics and hyperparameters
- Model comparison (accuracy, F1-score, inference time)
- Model registry (Production/Staging versions)

**4. Grafana Dashboard:**
- Service metrics (latency, requests, errors)
- Model performance over time
- System health indicators
- Custom alerts

> **Screenshot Placeholder:** Add GitHub repository structure

---

## Prerequisites

Before setting up the project, ensure you have:

- **Docker**: >= 20.10.0
- **Docker Compose**: >= 1.29.0 (or Docker Desktop with Compose V2)
- **Python**: 3.11
- **Git**: Latest version
- **GitHub Account**: For CI/CD setup

Optional but recommended:
- **make**: For simplified commands
- **curl** or **Postman**: For API testing

---

## Installation & Setup

### Clone Repository

```bash
git clone https://github.com/Fatma-Gaida/MLOps-Project.git
cd MLOps-Project
```

### Environment Setup

1. **Create a virtual environment:**

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Download the dataset:**

```bash
# Option 1: Using Kaggle API
kaggle datasets download -d https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset
unzip <dataset-file>.zip -d data/raw/


```

5. **Create `.env` file:**

```bash
cp .env.example .env
```

Then edit `.env` with your configuration:

```env
# FastAPI
MODEL_PATH=/transformer_model_final
LOG_LEVEL=info

# MLflow (optionnel)
MLFLOW_TRACKING_URI=http://mlflow:5000
```

### Docker Setup

1. **Build all services:**

```bash
docker-compose build
```

2. **Verify images were created:**

```bash
docker images | grep callcenterai
```

Expected output:
```
callcenterai-agent        latest    ...
callcenterai-tfidf        latest    ...
callcenterai-transformer  latest    ...
```

> **Screenshot Placeholder:** Add docker images output

---

## Project Phases

### Phase 1: Dataset Preparation

**Dataset:** Kaggle - IT Service Ticket Classification

**Specifications:**
- **Size:** ~47,000 tickets
- **Columns:** 
  - `Document`: Ticket text (customer request)
  - `Topic_group`: Category (target variable)
- **Categories:** Hardware, HR Support, Access, Miscellaneous, Storage, Purchase, etc.
- **Link:** [IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)

**Steps:**
1. Download dataset from Kaggle
2. Perform data cleaning (remove duplicates, handle missing values)
3. Split into train/test sets (80/20)


### Phase 2: Model Training

#### TF-IDF + SVM Model

**Approach:**
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classifier:** LinearSVC (Support Vector Machine)
- **Pipeline:** scikit-learn pipeline with TfidfVectorizer + LinearSVC
- **Calibration:** CalibratedClassifierCV for probability estimates

**Training Process:**
1. Create scikit-learn pipeline
2. Train on training set
3. Calibrate for probabilities
4. Log metrics to MLflow (accuracy, F1-score)
5. Save model artifacts

**Expected Performance:**
- Accuracy: ~85-90%
- Inference Time: <50ms
- Best for: Short, simple texts in English

> **Screenshot Placeholder:** Add TF-IDF+SVM training results from MLflow UI
> **Screenshot Placeholder:** Add confusion matrix for TF-IDF+SVM model

#### Transformer Model

**Approach:**
- **Model:** distilbert-base-multilingual-cased (Hugging Face)
- **Supports:** 104 languages (including FR, EN, AR)
- **Fine-tuning:** On ticket classification dataset
- **Framework:** Hugging Face Transformers library

**Training Process:**
1. Load pre-trained multilingual model
2. Fine-tune on ticket dataset
3. Log training metrics to MLflow
4. Save model and tokenizer
5. Register in MLflow model registry

**Expected Performance:**
- Accuracy: ~90-95%
- Inference Time: ~200-300ms
- Best for: Complex, multilingual texts

> **Screenshot Placeholder:** Add training loss curves over epochs
> **Screenshot Placeholder:** Add confusion matrix for Transformer model

### Phase 3: API Services

#### Service Overview

Three FastAPI microservices are created:
1. **TF-IDF/SVM Service** (Port 8002)
2. **Transformer Service** (Port 8001)
3. **AI Agent Service** (Port 8000)

**Why FastAPI?**
- Fast and efficient (based on Starlette + Pydantic)
- Easy to dockerize
- Built-in API documentation (Swagger UI)
- Prometheus metrics support
- Ideal for MLOps microservices

#### TF-IDF + SVM Service

**Features:**
- `/predict` endpoint for predictions
- `/health` endpoint for health checks
- `/metrics` endpoint for Prometheus
- Returns prediction with confidence scores

**Endpoints:**
- `POST /predict`: Get ticket classification
- `GET /health`: Service health check
- `GET /metrics`: Prometheus metrics

#### Transformer Service

**Features:**
- `/predict` endpoint for predictions
- `/health` endpoint for health checks
- `/metrics` endpoint for Prometheus
- Handles multilingual input
- Returns prediction with probability scores

**Endpoints:**
- `POST /predict`: Get ticket classification
- `GET /health`: Service health check
- `GET /metrics`: Prometheus metrics

#### AI Agent Service

**The Role of the AI Agent:**

The AI agent is a containerized microservice (deployed in Docker) that acts as an **intelligent chatbot**. Instead of simply responding, it:
- Receives customer tickets via REST API
- Cleans sensitive data (PII scrubbing)
- Analyzes the request (language, length, complexity)
- **Decides which model to use** (TF-IDF/SVM or Transformer)
- Returns prediction with confidence and explanation

**Key Features:**

**1. Intelligent Routing:**
- Analyzes text characteristics (length, complexity, language)
- Routes simple/short texts → TF-IDF/SVM (faster)
- Routes complex/multilingual texts → Transformer (more accurate)

**2. PII Scrubbing:**
- Removes/masks personal information before processing
- Protects: emails, phone numbers, credit cards, etc.
- Complies with data privacy regulations

**3. Confidence-Based Decisions:**
- Returns confidence scores with predictions
- Provides explanation for routing decisions
- Allows for threshold-based filtering

**Routing Logic:**
- **Text length < 20 words AND simple** → TF-IDF/SVM
- **Text length > 20 words OR multilingual OR complex** → Transformer

**Endpoints:**
- `POST /predict`: Classify ticket with intelligent routing
- `GET /health`: Service health check
- `GET /metrics`: Prometheus metrics

### Phase 4: Containerization

#### Docker Architecture

Each service has its own Dockerfile and all services are orchestrated with docker-compose.

**Services to Dockerize:**
1. TF-IDF/SVM service
2. Transformer service
3. AI Agent service
4. MLflow server
5. Prometheus
6. Grafana

#### Docker Compose Configuration

**File:** `docker-compose.yml`

**Services Defined:**
- `tfidf_svm`: TF-IDF model service (port 8002)
- `transformer`: Transformer model service (port 8001)
- `agent`: AI agent service (port 8000)
- `mlflow`: MLflow tracking server (port 5000)
- `prometheus`: Metrics collection (port 9090)
- `grafana`: Monitoring dashboards (port 3000)

**Networks:**
- All services connected via `mlops_network`
- Internal communication between services
- External access via exposed ports

**Volumes:**
- `mlflow_data`: Persist MLflow experiments
- `prometheus_data`: Persist metrics
- `grafana_data`: Persist dashboards

> **Screenshot Placeholder:** Add Docker Desktop showing all running containers


### Phase 6: Tests & Validation
#### Unit and Integration Tests
Each service includes unit tests located in:

- `services/tfidf_service/tests/test_api.py`
- `services/transformer_service/tests/test_api.py`

##### What they test
- **Health endpoints:** `/health` responds correctly.
- **Metrics endpoints:** `/metrics` exposes Prometheus metrics.
- **Prediction endpoints:** `/predict` returns expected categories, probabilities, and confidence.
- **Input validation:** handles empty, very long, multilingual, or special-character inputs.
- **Performance:** response times, batch predictions, benchmarks.
- **Model loading and categories:** ensures correct categories and model behavior.
- **Error handling:** invalid JSON, wrong HTTP methods, missing fields.


#### CI/CD Verification

**Checks in Pipeline:**
- All tests must pass
- Code coverage > 80%
- No linting errors
- No security vulnerabilities (Trivy scan passes)

> **Screenshot Placeholder:** Add pytest output showing all tests passing


---

## Project Components

### TF-IDF + SVM Model

**Description:**
Classic NLP approach using TF-IDF vectorization and Support Vector Machine classifier.

**Characteristics:**
- **Speed:** Very fast inference (<50ms)
- **Accuracy:** Good for standard texts (~85-90%)
- **Language:** Primarily English
- **Use Case:** High-volume, simple ticket classification

**Model Pipeline:**
1. TF-IDF Vectorizer (max_features=15000, ngram_range=(1,2))
2. LinearSVC classifier (C=1.0)
3. Calibration for probability estimates

**Files:**
- `models/tfidf_svm/model.pkl`: Trained model
- `models/tfidf_svm/vectorizer.pkl`: TF-IDF vectorizer
- `services/tfidf_svm/app.py`: FastAPI service

> **Screenshot Placeholder:** Add TF-IDF model performance metrics from MLflow

### Transformer Model

**Description:**
Advanced deep learning model using multilingual BERT variant for complex text classification.

**Characteristics:**
- **Speed:** Slower inference (~200-300ms)
- **Accuracy:** High accuracy (~90-95%)
- **Language:** Multilingual (104 languages)
- **Use Case:** Complex, multilingual, or nuanced tickets

**Model Details:**
- **Architecture:** distilbert-base-multilingual-cased
- **Parameters:** ~135M
- **Languages:** FR, EN, AR, and 101 others
- **Fine-tuned:** On ticket classification dataset

**Files:**
- `models/transformer/`: Model weights and config
- `models/transformer/tokenizer/`: Tokenizer files
- `services/transformer/app.py`: FastAPI service


### Intelligent AI Agent

**Description:**
Smart routing service that analyzes requests and directs them to the optimal model.

**Responsibilities:**

**1. Request Analysis:**
- Text length calculation
- Language detection
- Complexity assessment
- Special character analysis

**2. PII Protection:**
- Email masking: `user@email.com` → `[EMAIL]`
- Phone masking: `123-456-7890` → `[PHONE]`
- IP addresses: `123.96.14.78` → `[IP]`
- Credit card masking: `1234-5678-9012-3456` → `[CARD]`

**3. Intelligent Routing:**
- Simple/short → TF-IDF/SVM (fast)
- Complex/multilingual → Transformer (accurate)
- Confidence-based fallback

**4. Result Aggregation:**
- Combines prediction with routing explanation
- Returns confidence scores
- Provides transparency

**Decision Flow:**
```
Request → PII Scrubbing → Analysis → Routing Decision → Model Call → Response
```

### MLflow Tracking

**Purpose:**
Complete experiment tracking and model lifecycle management.

**Features:**

**1. Experiment Tracking:**
- Log parameters (hyperparameters, model config)
- Log metrics (accuracy, F1-score, training loss)
- Log artifacts (models, plots, datasets)
- Track multiple runs for comparison

**2. Model Registry:**
- Version control for models
- Stage management (None/Staging/Production)
- Model lineage and metadata
- Deployment readiness tracking

**3. Model Comparison:**
- Side-by-side experiment comparison
- Metric visualization
- Parameter impact analysis
- Best model selection

**Accessing MLflow:**
```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Access UI
http://localhost:5000
```

> **Screenshot Placeholder:** Add MLflow experiments list view

### Monitoring Stack

#### Prometheus

**Description:**  
Open-source monitoring and alerting toolkit for metrics collection.

**Configuration File:**  
`prometheus/prometheus.yml`

**Scraped Targets:**
- Agent Service: [http://agent_service:8000/metrics](http://agent_service:8000/metrics)
- TF-IDF Service: [http://tfidf_service:8002/metrics](http://tfidf_service:8002/metrics)
- Transformer Service: [http://transformer_service:8001/metrics](http://transformer_service:8001/metrics)

**Metrics Collected:**
- Request count per service
- Request duration (latency)
- Error rates
- Prediction counts by category
- System resources (CPU, memory)

**Accessing Prometheus:**  
```text
http://localhost:9090
```

**Example Queries:**
# Total requests over the last 5 minutes
sum(rate(requests_total[5m]))

# Average request latency
avg(request_duration_seconds)

# Error rate over the last 5 minutes
rate(errors_total[5m]) / rate(requests_total[5m])


> **Screenshot Placeholder:** Add Prometheus targets page showing all services UP
> **Screenshot Placeholder:** Add Prometheus graph showing request rate over time

#### Grafana

**Description:**
Visualization platform for creating dashboards from Prometheus metrics.

**Configuration:**
- Data source: Prometheus (http://prometheus:9090)
- Default credentials: admin/admin
- Dashboards in: `grafana/dashboards/`

**Accessing Grafana:**
```
http://localhost:3000
```

**Available Dashboards:**

**1. System Overview Dashboard:**
- Total requests per minute
- Active services status
- Overall error rate
- System resource usage

**2. Model Performance Dashboard:**
- Inference latency (p50, p95, p99)
- Predictions per model
- Model accuracy trends
- Confidence score distribution

**3. Agent Analytics Dashboard:**
- Routing decisions (TF-IDF vs Transformer)
- PII scrubbing statistics
- Request characteristics
- Agent latency breakdown

**4. Business Metrics Dashboard:**
- Tickets by category
- Classification trends
- Peak usage times
- SLA compliance

**Alert Configuration:**
- High error rate (>5%)
- Slow inference time (>2s)
- Service unavailability
- Low confidence predictions (>20%)

> **Screenshot Placeholder:** Add Grafana main dashboard overview
> **Screenshot Placeholder:** Add Model Performance dashboard with graphs
> **Screenshot Placeholder:** Add Agent Analytics dashboard
> **Screenshot Placeholder:** Add Grafana alert rules configuration

---

## Running the Application

### Local Development

**Step-by-Step Setup:**

**1. Start MLflow Server:**
```bash
mlflow server --host 0.0.0.0 --port 5000
```

**2. Start TF-IDF/SVM Service:**
```bash
cd services/tfidf_service
uvicorn app:app --host 0.0.0.0 --port 8002
```

**3. Start Transformer Service:**
```bash
cd services/transformer_service
uvicorn app:app --host 0.0.0.0 --port 8001
```

**4. Start AI Agent:**
```bash
cd services/agent_service
uvicorn app:app --host 0.0.0.0 --port 8000
```

**5. Start Prometheus:**
```bash
prometheus --config.file=prometheus/prometheus.yml
```

**6. Start Grafana:**
```bash
grafana-server --config=grafana/grafana.ini
```

**Service URLs:**
- AI Agent: http://localhost:8000
- TF-IDF Service: http://localhost:8002
- Transformer Service: http://localhost:8001
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### Docker Deployment

**Quick Start:**

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps

# Stop all services
docker-compose down
```

**Service Health Checks:**

```bash
# Check AI Agent
curl http://localhost:8000/health

# Check TF-IDF Service
curl http://localhost:8002/health

# Check Transformer Service
curl http://localhost:8001/health
```

---

## CI/CD Pipeline

Our CI/CD pipeline is implemented with **GitHub Actions** and automates code quality checks, testing, Docker image building, security scanning, and deployment. It runs on pushes, pull requests to `main` and `fatma`, and can also be triggered manually.

### Pipeline Overview

**Trigger Events:**  
- Push or Pull Request on `main` or `fatma`  
- Manual workflow dispatch  

**Environment Variables:**  
- `DOCKER_REGISTRY`: ghcr.io  
- `IMAGE_PREFIX`: `${{ github.repository_owner }}/callcenterai`  

### Jobs

#### 1. Lint & Code Quality
- Runs on all services (`agent_service`, `tfidf_service`, `transformer_service`)  
- Tools: **black**, **flake8**, **isort**, **bandit**  
- Checks for code formatting, linting issues, import order, and security vulnerabilities  

#### 2. Unit Tests
- Runs on all services after linting  
- Uses **pytest**, **pytest-cov**, **pytest-asyncio**, **httpx**  
- Coverage report uploaded to **Codecov**  
- Skips if no tests are found for a service  

#### 3. Build Docker Images
- Builds images for all services  
- Transformer service uses `no-cache` and disk cleanup due to size  
- Uses **Docker Buildx**, logs in to GitHub Container Registry  
- Creates `mlruns` directories to prevent empty folder errors  

#### 4. Security Scan
- Uses **Trivy** to scan `agent_service` and `tfidf_service` images  
- Generates SARIF reports for GitHub security tab  
- Outputs table report for critical/high vulnerabilities  

#### 5. Push Docker Images
- Pushes Docker images to GitHub Container Registry only for `main` branch  
- Uses cached builds when possible to save space  
- Metadata tagging includes branch, SHA, and `latest`  

#### 6. Notification
- Summarizes workflow results in Markdown format  
- Displays job results (Lint, Tests, Build, Security Scan, Push) and overall status ✅ / ❌  
- Adds a summary to GitHub Actions workflow summary  

### Tools & Technology
- **Python 3.11** for services  
- **Docker & Docker Buildx** for containerization  
- **GitHub Container Registry (GHCR)** for image storage  
- **Trivy** for security scanning  
- **Codecov** for coverage reporting  

### Notes
- Disk cleanup is critical for Transformer service during Docker build  
- Security scanning uses `CRITICAL,HIGH` severity  
- Workflow continues on errors for linting and tests but reports them in summary  

### Pipeline Workflow Diagram

```
┌─────────────┐
│  Code Push  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Linting   │ ─── Black, Flake8, isort
└──────┬──────┘
       │ ✓ Pass
       ▼
┌─────────────┐
│  Security   │ ─── Bandit, Trivy
└──────┬──────┘
       │ ✓ Pass
       ▼
┌─────────────┐
│   Testing   │ ─── pytest (unit + integration)
└──────┬──────┘
       │ ✓ Pass
       ▼
┌─────────────┐
│    Build    │ ─── Docker images
└──────┬──────┘
       │ ✓ Success
       ▼
┌─────────────┐
│     Push    │ ─── Docker Hub / ghcr.io
└──────┬──────┘
       │ ✓ Success
       ▼
┌─────────────┐
│   Deploy    │ ─── Production (main branch only)
└──────┬──────┘
       │ ✓ Success
       ▼
┌─────────────┐
│   Notify    │ ─── Slack, Email
└─────────────┘
```

> **Screenshot Placeholder:** Add complete GitHub Actions workflow visualization

---

## API Documentation

### API Endpoints

#### AI Agent Service (Port 8000)

**Base URL:** `http://localhost:8000`

**Endpoints:**

**1. Classify Ticket**
- **Method:** POST
- **Path:** `/predict`
- **Description:** Classify a customer ticket with intelligent routing
- **Authentication:** None (add if needed)

**2. Health Check**
- **Method:** GET
- **Path:** `/health`
- **Description:** Check if service is healthy

**3. Metrics**
- **Method:** GET
- **Path:** `/metrics`
- **Description:** Prometheus metrics endpoint

#### TF-IDF/SVM Service (Port 8001)

**Base URL:** `http://localhost:8002`

**Endpoints:**

**1. Predict**
- **Method:** POST
- **Path:** `/predict`
- **Description:** Get prediction from TF-IDF model

**2. Health Check**
- **Method:** GET
- **Path:** `/health`

**3. Metrics**
- **Method:** GET
- **Path:** `/metrics`

#### Transformer Service (Port 8002)

**Base URL:** `http://localhost:8001`

**Endpoints:**

**1. Predict**
- **Method:** POST
- **Path:** `/predict`
- **Description:** Get prediction from Transformer model

**2. Health Check**
- **Method:** GET
- **Path:** `/health`

**3. Metrics**
- **Method:** GET
- **Path:** `/metrics`


### Request/Response Examples

#### Example 1: Simple Ticket (Routes to TF-IDF)

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My laptop screen is broken"
  }'
```

**Response:**
```json
{
  "prediction": "Hardware",
  "confidence": 0.92,
  "model_used": "TF-IDF + SVM",
  "routing_reason": "Short and simple text - using fast model",
  "text_cleaned": "My laptop screen is broken"
}
```

#### Example 2: Complex Multilingual Ticket (Routes to Transformer)

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bonjour, je ne peux pas accéder à mon compte depuis hier. J'\''ai essayé de réinitialiser mon mot de passe mais je n'\''ai pas reçu l'\''email de confirmation. Pouvez-vous m'\''aider?"
  }'
```

**Response:**
```json
{
  "prediction": "Access",
  "confidence": 0.88,
  "model_used": "Transformer",
  "routing_reason": "Complex or multilingual text - using advanced model",
  "text_cleaned": "Bonjour, je ne peux pas accéder à mon compte depuis hier. J'ai essayé de réinitialiser mon mot de passe mais je n'ai pas reçu l'email de confirmation. Pouvez-vous m'aider?"
}
```

#### Example 3: Health Check

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "ai_agent",
  "timestamp": "2025-12-10T10:30:00Z",
  "version": "1.0.0",
  "dependencies": {
    "tfidf_service": "UP",
    "transformer_service": "UP"
  }
}
```

> **Screenshot Placeholder:** Add Postman collection showing all API examples

---

## Troubleshooting

### Common Issues

#### 1. Docker Containers Won't Start

**Symptoms:**
- Services fail to start
- Containers exit immediately
- Port conflicts

**Solutions:**
```bash
# Check logs for errors
docker-compose logs [service_name]

# Check port availability
lsof -i :8000
lsof -i :8001
lsof -i :8002

# Rebuild containers
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d

# Check Docker resources
docker system df
docker system prune  # Clean up if needed
```

#### 2. MLflow Connection Errors

**Symptoms:**
- Can't connect to MLflow UI
- Models fail to log experiments
- Tracking URI errors

**Solutions:**
```bash
# Verify MLflow is running
curl http://localhost:5000/health

# Check MLflow logs
docker-compose logs mlflow

# Verify environment variable
echo $MLFLOW_TRACKING_URI

# Restart MLflow service
docker-compose restart mlflow
```

#### 3. Model Prediction Errors

**Symptoms:**
- 500 errors from model services
- Slow inference times
- Wrong predictions

**Solutions:**
```bash
# Check model files exist
ls -la models/tfidf_server/
ls -la models/transformer_server/

# Check service logs
docker-compose logs tfidf_server
docker-compose logs transformer_server

# Verify model loaded correctly
curl http://localhost:8001/health
curl http://localhost:8002/health

# Check memory usage
docker stats
```

#### 4. Agent Routing Issues

**Symptoms:**
- Agent always uses same model
- Routing logic not working
- PII not being scrubbed

**Solutions:**
```bash
# Check agent logs
docker-compose logs agent

# Test with different inputs
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "short text"}'

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "very long complex multilingual text..."}'

# Verify model services are reachable
docker-compose exec agent curl http://tfidf_server:8002/health
docker-compose exec agent curl http://transformer_server:8001/health
```

#### 5. Prometheus Not Scraping Metrics

**Symptoms:**
- No data in Grafana
- Targets show as DOWN in Prometheus
- Missing metrics

**Solutions:**
```bash
# Check Prometheus targets
# Open: http://localhost:9090/targets

# Verify services expose /metrics endpoint
curl http://localhost:8000/metrics
curl http://localhost:8001/metrics
curl http://localhost:8002/metrics

# Check Prometheus config
cat prometheus/prometheus.yml

# Restart Prometheus
docker-compose restart prometheus
```

#### 6. Grafana Dashboards Not Working

**Symptoms:**
- Dashboards show no data
- Data source connection failed
- Panels show errors

**Solutions:**
```bash
# Verify Prometheus data source
# Grafana → Configuration → Data Sources

# Test Prometheus connection
curl http://prometheus:9090/api/v1/query?query=up

# Check Grafana logs
docker-compose logs grafana

# Re-import dashboards
# Grafana → Dashboards → Import
# Upload from grafana/dashboards/

# Verify time range in dashboard
# Set to "Last 15 minutes" or "Last 1 hour"
```

#### 7. CI/CD Pipeline Failures

**Symptoms:**
- GitHub Actions workflow fails
- Tests don't pass
- Docker build fails

**Solutions:**
```bash
# Check GitHub Actions logs
# Go to Actions tab in repository

# Run linting locally
black --check .
flake8 .
isort --check-only .

# Build Docker images locally
docker-compose build

# Check secrets are configured
# GitHub → Settings → Secrets and variables → Actions
```

### Debug Mode

**Enable verbose logging:**

```bash
# Set in .env file
LOG_LEVEL=DEBUG

# Restart services
docker-compose down
docker-compose up -d

# View detailed logs
docker-compose logs -f
```

### Getting Help

**Resources:**
- Check logs first: `docker-compose logs [service]`
- Check GitHub Issues for similar problems
- Contact maintainers (see contact section)

---

## Contributing

We welcome contributions to CallCenterAI! Here's how you can help:

### Development Workflow

1. **Fork the repository**
```bash
# Click "Fork" on GitHub
```

2. **Clone your fork**
```bash
git clone https://github.com/your-username/Fatma-Gaida/MLOps-Project.git
cd MLOps-Project
```

3. **Create a feature branch**
```bash
git checkout -b feature/amazing-feature
```

4. **Make your changes**
- Write code
- Add tests
- Update documentation


5. **Commit changes**
```bash
git add .
git commit -m "Add amazing feature"
```

6. **Push to your fork**
```bash
git push origin feature/amazing-feature
```

7. **Open a Pull Request**
- Go to original repository on GitHub
- Click "New Pull Request"
- Select your branch
- Describe your changes

### Code Style Guidelines

**Python:**
- Follow PEP 8
- Use type hints
- Write docstrings for functions
- Maximum line length: 88 characters (Black default)

**Imports:**
- Use isort for import sorting
- Group imports: standard library, third-party, local

**Testing:**
- Write tests for new features
- Maintain coverage > 80%
- Use descriptive test names

### Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure CI/CD passes** (all checks green)
4. **Request review** from maintainers
5. **Address feedback** promptly
6. **Squash commits** if requested

### Areas for Contribution

**Features:**
- Additional NLP models
- More sophisticated routing logic
- API authentication
- Rate limiting
- Caching layer

**Improvements:**
- Performance optimization
- Better error handling
- Enhanced monitoring
- Additional tests

**Documentation:**
- Tutorial videos

## Contributing

We welcome contributions! Please follow these guidelines:

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for functions
- Add tests for new features

### Pull Request Process

1. Update README if needed
2. Add tests for new functionality
3. Ensure CI/CD pipeline passes
4. Request review from maintainers
5. Address review comments

---

## Contact & Support

**Project Maintainers:**
- Fatma Zahra Gaid - [fatmazahragaida@gmail.com](fatmazahragaida@gmail.com)
- Harzali Chaima - [chaimaharzali333@gmail.com](mailto:chaimaharzali333@gmail.com)

**Project Link:** [https://github.com/Fatma-Gaida/MLOps-Project](https://github.com/Fatma-Gaida/MLOps-Project)


---

## Acknowledgments

- MLflow for experiment tracking
- Streamlit for the UI framework
- Prometheus & Grafana for monitoring
- Docker for containerization
- GitHub Actions for CI/CD

---

**Last Updated:** December 2025
**Version:** 1.0.0
