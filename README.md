# LLM Diagnosis Toolkit

- [LLM Diagnosis Toolkit](#llm-diagnosis-toolkit)
- [Basic Information](#basic-information)
- [Introduction](#introduction)
- [Research Summary](#research-summary)
- [Keywords](#keywords)
- [Architecture Overview](#architecture-overview)
- [Data](#data)
- [Dependencies](#dependencies)
- [Installation & Usage](#installation--usage)
  - [Quick Start (VM Deployment)](#quick-start-vm-deployment)
  - [Local Development](#local-development)
  - [Distributed Setup](#distributed-setup)
- [Project Structure](#project-structure)
- [License](#license)

## Introduction

The LLM Diagnosis Toolkit is a web-based platform designed to support practitioners in systematically diagnosing undesirable behavior in large language models (LLMs). While existing Explainable AI (XAI) methods provide valuable insights into model internals, they often fall short in helping practitioners identify *why* a model fails in specific situations. This toolkit bridges that gap by providing a structured diagnostic framework that connects observable symptoms to investigative strategies and potential underlying causes.

Built for researchers and practitioners who work with transformer-based language models, the toolkit operationalizes a diagnostic process that goes beyond explainability to enable hypothesis-driven investigation of model failures. The platform features a distributed architecture that separates lightweight user-facing components from computationally intensive model processing, enabling flexible deployment across different hardware configurations.

## Research Summary

**Research Question**: What kinds of tooling support practitioners in diagnosing undesirable behavior in large language models?

**Sub-Questions**:

1. What challenges and limitations shape how (large) language models are diagnosed?
2. What kinds of diagnostic support help practitioners identify and understand failure patterns?
3. How does the toolkit complement or improve upon current diagnostic practices used by practitioners?

**Approach**: This thesis follows a Design Science Research methodology, informed by practitioner insights from a [formative study](https://repository.tudelft.nl/record/uuid:700ce316-5047-4182-ae25-e47db8ca4054) conducted at a software company. The study revealed that formal explanation tools were infrequently used during real debugging scenarios, with practitioners emphasizing the need for contextual information and structured guidance to reason about failure causes rather than isolated interpretability artifacts.

**Implementation**: The toolkit consists of two main components:

- **Navigation Service**: A Django-based web interface that provides the user-facing diagnostic workflow
- **Processing Service**: A FastAPI-based backend that handles model loading, inference, and diagnostic computations using GPU acceleration

The architecture supports two deployment modes:

1. **Local**: All-in-one setup for development and testing on a single machine
2. **Distributed**: Separation of frontend and backend for flexible resource utilization on different hardware (e.g., laptop + GPU server)

**Results**: TBD (to be filled in after evaluation)

## Keywords

- LLM Diagnosis
- Model Debugging
- NLP Toolkit
- Transformer Models

## Architecture Overview

The toolkit uses a modular monolitic / macroservice architecture consisting of two main services: the Navigation Service (Django) and the Processing Service (FastAPI). This design allows for flexible deployment options, enabling users to run the entire toolkit on a single machine or distribute the services across multiple machines based on resource availability.

```text
┌─────────────────────────────────────────────────────────┐
│                    User's Browser                       │
│              http://<vm-or-laptop>:8000                 │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Navigation Service (Django)                │
│  - Web UI for diagnostic workflows                      │
│  - SQLite database for user sessions                    │
│  - Lightweight, runs on any machine                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│             Processing Service (FastAPI)                │
│  - HuggingFace model loading & inference                │
│  - GPU-accelerated computation                          │
│  - MongoDB for datasets & results                       │
│  - Requires powerful hardware (GPU recommended)         │
└─────────────────────────────────────────────────────────┘
```

## Data

All datasets used by the toolkit are automatically loaded from HuggingFace Hub during runtime. No external data hosting is required. The toolkit includes seed data for pre-selected diagnostic tests and model configurations.

Test results and model outputs are stored in a MongoDB database (processing service) and user-specific session data is stored in a SQLite database (navigation service). This separation ensures efficient data management and retrieval during diagnostic workflows, while keeping user session data lightweight.

## Dependencies

**Core Technologies**:

- **Docker & Docker Compose**: Container orchestration
- **NVIDIA Container Toolkit**: GPU access in Docker (optional, for GPU support)
- **Python 3.13**: Runtime environment
- **uv**: Python package manager

**Python Frameworks**:

- **Django 5.1+**: Web framework (navigation service)
- **FastAPI**: API framework (processing service)
- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace model library
- **MongoDB**: NoSQL database (processing data)
- **SQLite**: Relational database (navigation data)

**Full Dependencies**: See `code/navigation/pyproject.toml` and `code/processing/pyproject.toml` for complete package lists with versions.

**Configuration Files**:

- `.env` - Local development configuration
- `.env.vm` - VM deployment configuration  
- `.env.processing` - Distributed processing service configuration
- `.env.navigation` - Distributed navigation service configuration

## Installation & Usage

The toolkit supports three deployment scenarios. Choose the one that fits your use case:

### Quick Start (VM Deployment)

For running the complete toolkit on a single server:

```bash
# 1. Clone the repository
git clone https://github.com/delftcrowd/llm_diagnosis_toolkit.git
cd llm_diagnosis_toolkit/code

# 2. Pull large files for diagnostic seed data
cd processing/app/data
git lfs pull  # Pull large files for seed data
cd ../../..

# 3. Configure environment
cp .env.vm.example .env.vm
nano .env.vm  # Update credentials and URLs

# 4. Start services
docker compose -f docker-compose.vm.yml --env-file .env.vm -p "dtk" build
docker compose -f docker-compose.vm.yml --env-file .env.vm -p "dtk" up -d

# 5. Seed databases
cd scripts
sh seed_dbs.sh

# 6. Access at http://localhost:8000
```

**Detailed instructions**: See [code/VM_SETUP.md](code/VM_SETUP.md)

### Local Development

For development and testing on a single machine:

```bash
# 1. Clone and configure
git clone https://github.com/delftcrowd/llm_diagnosis_toolkit.git
cd llm_diagnosis_toolkit/code
cp .env.example .env  # Create and configure

# 2. Start all services
docker compose --profile all -p "dtk_dev" up -d

# 3. Seed databases
cd scripts
sh seed_dbs.sh

# 4. Access at http://localhost:8000
```

### Distributed Setup

For separating frontend (laptop) from backend (GPU server):

**On Processing Server**:

```bash
cd code
cp .env.processing.example .env.processing  # Configure
docker compose -f docker-compose.processing.yml --env-file .env.processing -p "dtk_processing" up -d
cd scripts
sh seed_processing.sh
```

**On Navigation Client**:

```bash
cd code
cp .env.navigation.example .env.navigation  # Configure with processing server URL
docker compose -f docker-compose.navigation.yml --env-file .env.navigation -p "dtk_navigation" up -d
cd scripts
sh seed_navigation.sh
```

**Detailed instructions**: See [code/DISTRIBUTED_SETUP.md](code/DISTRIBUTED_SETUP.md)

## Project Structure

```bash
llm_diagnosis_toolkit/
│
├── README.md                    # This file
├── LICENSE                      # Apache 2.0 License
├── template.md                  # README template
│
├── code/                        # Main application code
│   ├── docker-compose.yml       # Local all-in-one deployment
│   ├── docker-compose.vm.yml    # VM deployment
│   ├── docker-compose.processing.yml  # Distributed: processing only
│   ├── docker-compose.navigation.yml  # Distributed: navigation only
│   ├── .env.vm                  # VM environment template
│   ├── .env.processing          # Processing service template
│   ├── .env.navigation          # Navigation service template
│   ├── VM_SETUP.md              # VM deployment guide
│   ├── DISTRIBUTED_SETUP.md     # Distributed deployment guide
│   │
│   ├── navigation/              # Django web interface
│   │   ├── Dockerfile.multistage
│   │   ├── pyproject.toml       # Python dependencies
│   │   ├── manage.py            # Django management
│   │   └── projects/            # Main Django app
│   │
│   ├── processing/              # FastAPI backend
│   │   ├── Dockerfile.multistage      # CPU version
│   │   ├── Dockerfile.multistage.gpu  # GPU version
│   │   ├── pyproject.toml       # Python dependencies
│   │   ├── mongo-init.js        # MongoDB initialization
│   │   └── app/                 # FastAPI application
│   │       ├── main.py
│   │       ├── controllers/     # API endpoints
│   │       ├── models/          # Data models
│   │       ├── services/        # Business logic
│   │       └── database/        # DB seeders
│   │
│   └── scripts/                 # Deployment scripts
│       ├── seed_dbs.sh          # Seed both databases (local/VM)
│       ├── seed_processing.sh   # Seed processing DB (distributed)
│       └── seed_navigation.sh   # Seed navigation DB (distributed)
│
└── report/                      # Thesis report and presentations
    └── README.md
```

**Key Directories**:

- `code/navigation/`: User-facing web interface built with Django
- `code/processing/`: GPU-accelerated backend for model operations
- `code/scripts/`: Database seeding and deployment automation
- `report/`: Academic documentation and thesis materials

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
