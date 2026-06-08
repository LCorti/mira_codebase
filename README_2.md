# MIRA codebase

MIRA is a toolkit designed for diagnosing large language models (LLMs). It provides tools and workflows to inspect, evaluate, and troubleshoot LLM behavior, making it easier for practitioners to understand model outputs and improve performance.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python**: Required for running the application and installing dependencies.  
    [Install Python 3.15](https://www.python.org/downloads/)
- **Docker**: Used to containerize and run the application.  
    [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: Orchestrates multi-container Docker applications.  
    [Install Docker Compose](https://docs.docker.com/compose/install/)
- Git LFS: Required for pulling large files used in seed data.  
    [Install Git LFS](https://git-lfs.com/)

> **Note:**  
> Make sure Python 3.15, Docker and Docker Compose are running and accessible from your terminal.  
> You can verify your installations with:
>
> ```sh
> python --version
> docker --version
> docker-compose --version
> git lfs version
> ```
> Depending on the version of docker compose installed, you will need to use 
> - ```docker-compose``` for v1 (deprecated)
> - ```docker compose``` for v2 (current)
>

## Python Dependencies

Install the required Python packages using pip:

```sh
pip install -r requirements.txt
```

This will install the following dependencies:

- Django 5.2.7
- pymongo 4.15.2

## Local Setup

Running all services on a single machine.

### Pull Large Files for Diagnostic Seed Data

Navigate to the data directory and pull large files using Git LFS:

```sh
cd processing/app/data
git lfs pull  # Pull large files for seed data
cd ../../..
```

### Environment File

To configure environment variables, copy the example environment file and update it as needed:

```sh
cp .env.example .env
```

Open the newly created `.env` file and update the variable values to match your setup.

### Build the Services

First time or after changes to Dockerfiles/dependencies:

```sh
docker-compose -f docker-compose.yml --env-file .env --profile all -p "dtk" build --no-cache
```

### Run the Services

Start all services:

```sh
docker-compose -f docker-compose.yml --env-file .env --profile all -p "dtk" up
```

Or start specific services:

```sh
# Navigation only
docker-compose -f docker-compose.yml --env-file .env --profile navigation -p "dtk" up

# Processing only
docker-compose -f docker-compose.yml --env-file .env --profile processing -p "dtk" up
```

### Seed the Database

With services running, open a new terminal:

```sh
cd code/scripts

# MacOS/Linux
sh seed_dbs.sh

# Windows
seed_dbs.bat
```

⚠️ The warnings `Failed to sync method <id> after creation` can be ignored.

## Distributed Setup

Run services across multiple machines (e.g., navigation on laptop, processing on desktop/VM with more compute resources).

For detailed instructions including network configuration, firewall setup, and troubleshooting, see [DISTRIBUTED_SETUP.md](documentation/DISTRIBUTED_SETUP.md).

### Pull Large Files

Like in the local setup, navigate to the data directory and pull large files using Git LFS:

```sh
cd processing/app/data
git lfs pull  # Pull large files for seed data
cd ../../..
```

### Environment File Distributed Setup

See [DISTRIBUTED_SETUP.md](documentation/DISTRIBUTED_SETUP.md) for detailed instructions on setting up environment files for distributed setups.

Open the newly created `.env.processing` and `.env.navigation` files and update the variable values to match your setup.

### Quick Start

**On Processing Machine (Desktop/VM):**

```sh
# Build
docker-compose -f docker-compose.processing.yml --env-file .env.processing -p "dtk" build --no-cache

# Run
docker-compose -f docker-compose.processing.yml --env-file .env.processing -p "dtk" up

# Seed (in separate terminal)
cd code/scripts
sh seed_processing.sh
```

**On Navigation Machine (Laptop):**

```sh
# Build
docker-compose -f docker-compose.navigation.yml --env-file .env.navigation -p "dtk" build --no-cache

# Run
docker-compose -f docker-compose.navigation.yml --env-file .env.navigation -p "dtk" up

# Seed (in separate terminal)
cd code/scripts
sh seed_navigation.sh
```
