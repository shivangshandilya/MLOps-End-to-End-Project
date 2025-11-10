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

Let's see the how the project was created, step by step (a brief overview):
------

### 1. Model Training

We chose [IRIS Classification Dataset](https://www.kaggle.com/datasets/uciml/iris) for this project, we chose to train the model on 5 different algorithms being Logistic Regression, Decision Trees, Random Forest, SVM and XGBoost. From the model being fit via these diff algorithms we compared the results and saw which algorithm best fits the model. You might ask how we compared the results, we did so using Weights & Biases in the next step.

### 2. Data Visualizaton

We made use of [Weights & Biases](https://wandb.ai/site/) to visualize our data and model via different algorithms, we made scatter and bar plots to understand our data better, the pic is shown below. Wandb makes it easier to plot and compare metrics.

<img width="1819" height="480" alt="Screenshot 2025-11-02 235002" src="https://github.com/user-attachments/assets/1a1403dd-9b39-46bc-adad-6c8f01fa4081" />

<img width="1836" height="434" alt="Screenshot 2025-11-03 011125" src="https://github.com/user-attachments/assets/06910d05-6175-4145-9113-327fbb5e5bb6" />

We were easily able to compare the metrics of our model fitted with different algorithms, we could compare the model like below in the picture and even individually with their respective metric plots

<img width="1919" height="936" alt="Screenshot 2025-11-03 012008" src="https://github.com/user-attachments/assets/4a932a11-14fc-4227-af02-0b55c1ea13e6" />

### 3. Data Versioning

We used [DVC](https://dvc.org/) for versioning of our models and data, it was another tool that I worked with for the very first time. DVC works alongside Git and helps track large files, data pipelines, and model checkpoints without actually storing them in the Git repository. But in this experiment I stored DVC logs in a local directory rather than a remote one like Amazon S3. 

### 4. Containerization

Now the next step involved building an image for our model now we can't deploy our model straight up we needed to package it into an app and so we did exactly that we created a [FastAPI](https://fastapi.tiangolo.com/) backend for our app with various endpoints and even a /metrics endpoint to expose our metrics to [Prometheus](https://prometheus.io/) in future step.

<img width="1919" height="938" alt="Screenshot 2025-11-04 170156" src="https://github.com/user-attachments/assets/38dfb663-d14f-498a-9a5c-b3e1e6009379" />
<img width="1914" height="937" alt="Screenshot 2025-11-04 170213" src="https://github.com/user-attachments/assets/58d20f50-df70-43dc-ad3c-723fca02013d" />

### 5. Container Registry

The next step involves pushing our Docker Image to the AWS ECR, now we dont do this manually coz where is the fun in that we leverage the use of CI/CD to accomplish this task, we have our [publish-aws-ecr](https://github.com/shivangshandilya/MLOps-End-to-End-Project/blob/main/.github/workflows/publish-aws-ecr.yml) Github workflow which solves this it logs in to AWS ECR and then pushes our image to the registry.

<img width="1917" height="482" alt="Screenshot 2025-11-06 181125" src="https://github.com/user-attachments/assets/41189dcd-060d-4ec9-bd87-ff902e46ac73" />

### 6. Deployment

For deploying our application we leverage the use of AWS EKS, which fetches our image from the ECR. This is also accomplished by our GitHub Actions, after every push in the reposiory a new image is created automatically and is then pushed to ECR which is then fetched by EKS to keep our app up-to-date with any new changes.

<img width="1919" height="518" alt="Screenshot 2025-11-08 122735" src="https://github.com/user-attachments/assets/d8f38229-839c-4712-82e6-a59b256a7fa1" />

### 7. CI/CD pipeline

As discussed above the CI part of our pipeline is handled by GitHub Actions, which solves 2 crucial tasks: 1. It builds and tests our app for any errors and 2. as discussed above logs in to ECR to push our latest image and then apply manifest yaml files to our EKS cluster.

**Github Actions**
<img width="1919" height="805" alt="Screenshot 2025-11-06 181904" src="https://github.com/user-attachments/assets/b4ff017a-b1d4-4619-ab13-698d7c71c248" />

And the CD part of our pipeline is handled by ArgoCD which continuously monitors the health of our K8s cluster

**ArgoCD Dashboard for our app**
<img width="1919" height="938" alt="Screenshot 2025-11-08 135209" src="https://github.com/user-attachments/assets/9f17b121-82cf-494b-9560-4d60692eea4a" />

### 8. Monitoring & Observability

For this step we leveraged use of [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/)



