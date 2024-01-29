# GCP Vertex AI 


# Topics
- Basic model muilding - Notebook
- custom Docker environment
- Execute model script
- Deploy pipeline
- Register model
- Train and inference(Serving) pipeline
- Hyper-Parameter tuning 
- Distributed Training
- Model performance monitoring
- Data drift monitoring
- Retrain and model freshness


# Steps from development to deploy
- Save working pynb file (Notebook)
- Convert notebook to script
- modify script to directly communicate with the GCP file storage, with no reference to local path

# DOCKER CREATION PROCESS
## dependency is on Artifact Registry API - Service on GCP to be enabled
- PROJECT_ID='kubeflow-started'
- REPO_NAME='ml-repo-container'
- IMAGE_URI=us-central1-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/basic-ml-env-image:latest
- docker build ./ -t $IMAGE_URI
- docker push $IMAGE_URI
