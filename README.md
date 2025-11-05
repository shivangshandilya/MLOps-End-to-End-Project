# MLOps-End-to-End-Project

This project demonstrates a complete end-to-end MLOps workflow, covering the full lifecycle of model development, deployment, and monitoring:

- **Model Training** – Build and train the machine learning model
- **Experiment Tracking & Visualization** – Use Weights & Biases for logging metrics and comparing runs
- **Data Versioning** – Manage datasets and reproducibility using DVC
- **Containerization** – Package the model and code into a Docker image
- **Container Registry** – Push the Docker image to AWS ECR
- **Deployment** – Deploy the model on AWS EKS (Kubernetes)
- **CI/CD Pipeline** – Automate building and deployment using GitHub Actions + ArgoCD
- **Monitoring & Observability** – Track performance and system health using Prometheus and Grafana

<img width="1456" height="702" alt="image" src="https://github.com/user-attachments/assets/2901ca41-1a6d-4820-824a-7bd34b4ff7b7" />

Let's see the how the project was created, step by step:
------

## 1. Model Training

We chose [IRIS Classification Dataset](https://www.kaggle.com/datasets/uciml/iris) for this project, we chose to train the model on 5 different algorithms being Logistic Regression, Decision Trees, Random Forest, SVM and XGBoost. From the model being fit via these diff algorithms we compared the results and saw which algorithm best fits the model. You might ask how we compared the results, we did so using Weights & Biases in the next step.
