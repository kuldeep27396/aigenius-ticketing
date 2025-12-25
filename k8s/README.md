# Kubernetes Deployment Guide

## Overview

Deploy AIGenius Ticketing to any Kubernetes cluster (AWS EKS, Google GKE, Azure AKS, DigitalOcean Kubernetes, etc.).

## Prerequisites

- Kubernetes cluster (v1.24+)
- `kubectl` configured to access your cluster
- Container registry access (Docker Hub, GHCR, ECR, GCR, etc.)
- Ingress controller (optional, for external access)

## Quick Start

### 1. Create Namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

### 2. Create Secrets

```bash
# Copy the template
cp k8s/secrets.yaml.template k8s/secrets.yaml

# Edit with your actual values
nano k8s/secrets.yaml

# Apply secrets
kubectl apply -f k8s/secrets.yaml -n aigenius
```

### 3. Build and Push Container Image

```bash
# Using Docker
docker build -t your-registry/aigenius-ticketing:latest .
docker push your-registry/aigenius-ticketing:latest

# Using UV (recommended)
uv docker build -t your-registry/aigenius-ticketing:latest .
```

### 4. Update Image in Deployment

Edit `k8s/deployment.yaml` and update:
```yaml
image: your-registry/aigenius-ticketing:latest
```

Or use Kustomize to patch the image:
```bash
kustomize edit set image your-registry/aigenius-ticketing:latest
```

### 5. Deploy

```bash
# Apply all manifests
kubectl apply -f k8s/ -n aigenius

# Or using Kustomize
kubectl apply -k k8s/
```

### 6. Verify Deployment

```bash
# Check pods
kubectl get pods -n aigenius

# Check services
kubectl get svc -n aigenius

# Check logs
kubectl logs -f deployment/aigenius-ticketing -n aigenius

# Port forward for testing
kubectl port-forward svc/aigenius-ticketing 8000:80 -n aigenius
```

## Platform-Specific Setup

### AWS EKS

```bash
# Create cluster
eksctl create cluster --name aigenius --region us-east-1

# Create ECR repository
aws ecr create-repository --repository-name aigenius-ticketing

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t <account-id>.dkr.ecr.us-east-1.amazonaws.com/aigenius-ticketing:latest .
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/aigenius-ticketing:latest

# Update deployment image
image: <account-id>.dkr.ecr.us-east-1.amazonaws.com/aigenius-ticketing:latest
```

### Google GKE

```bash
# Create cluster
gcloud container clusters create aigenius --num-nodes=2 --zone us-central1-a

# Get credentials
gcloud container clusters get-credentials aigenius --zone us-central1-a

# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/aigenius-ticketing:latest .

# Update deployment image
image: gcr.io/PROJECT_ID/aigenius-ticketing:latest
```

### Azure AKS

```bash
# Create resource group and cluster
az group create --name aigenius-rg --location eastus
az aks create --resource-group aigenius-rg --name aigenius --node-count 2

# Get credentials
az aks get-credentials --resource-group aigenius-rg --name aigenius

# Build and push to ACR
az acr create --resource-group aigenius-rg --name aigeniusacr --sku Basic
az acr login --name aigeniusacr
docker build -t aigeniusacr.azurecr.io/aigenius-ticketing:latest .
docker push aigeniusacr.azurecr.io/aigenius-ticketing:latest

# Update deployment image
image: aigeniusacr.azurecr.io/aigenius-ticketing:latest
```

### DigitalOcean Kubernetes

```bash
# Create cluster
doctl kubernetes cluster create aigenius --region nyc1 --node-pool "name=aigenius-pool;count=2;size=s-2vcpu-4gb"

# Get credentials
doctl kubernetes cluster kubeconfig save aigenius

# Build and push to Docker Hub
docker build -t your-dockerhub-user/aigenius-ticketing:latest .
docker push your-dockerhub-user/aigenius-ticketing:latest

# Update deployment image
image: your-dockerhub-user/aigenius-ticketing:latest
```

## Ingress Setup

### Using NGINX Ingress Controller

```bash
# Install NGINX Ingress
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Install Cert-Manager for SSL
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

# Update ingress.yaml with your domain
# Then apply
kubectl apply -f k8s/ingress.yaml -n aigenius
```

## Monitoring

### View Logs

```bash
# All pods
kubectl logs -f -l app=aigenius-ticketing -n aigenius

# Specific pod
kubectl logs -f pod/aigenius-ticketing-xxxxx -n aigenius
```

### Port Forward for Local Testing

```bash
kubectl port-forward svc/aigenius-ticketing 8000:80 -n aigenius
```

### Scale Deployment

```bash
# Manual scaling
kubectl scale deployment aigenius-ticketing --replicas=5 -n aigenius

# HPA handles automatic scaling based on CPU/Memory
```

## Updating the Application

```bash
# Build new image
docker build -t your-registry/aigenius-ticketing:v1.0.1 .
docker push your-registry/aigenius-ticketing:v1.0.1

# Update deployment (rolling update)
kubectl set image deployment/aigenius-ticketing aigenius-ticketing=your-registry/aigenius-ticketing:v1.0.1 -n aigenius

# Or edit and apply
kubectl edit deployment aigenius-ticketing -n aigenius
```

## Troubleshooting

### Pod Not Starting

```bash
# Describe pod for events
kubectl describe pod <pod-name> -n aigenius

# Check logs
kubectl logs <pod-name> -n aigenius

# Get previous container logs if it crashed
kubectl logs <pod-name> --previous -n aigenius
```

### Database Connection Issues

```bash
# Verify secret exists
kubectl get secret aigenius-secrets -n aigenius -o yaml

# Check database connectivity from pod
kubectl exec -it <pod-name> -n aigenius -- nc -zv <db-host> 5432
```

### HPA Not Scaling

```bash
# Check HPA status
kubectl get hpa -n aigenius

# Describe HPA for conditions
kubectl describe hpa aigenius-ticketing -n aigenius
```

## Cleanup

```bash
# Delete all resources
kubectl delete -k k8s/

# Or individually
kubectl delete namespace aigenius
```

## Production Considerations

1. **Resource Limits**: Adjust CPU/memory based on actual usage
2. **Replicas**: Start with 2-3 replicas for high availability
3. **Database**: Use managed PostgreSQL (RDS, Cloud SQL, etc.)
4. **Secrets**: Use sealed-secrets or external secret managers (Vault, AWS Secrets Manager)
5. **Monitoring**: Add Prometheus/Grafana for metrics
6. **Logging**: Add ELK/Loki for centralized logging
7. **Backups**: Regular database backups
8. **SSL/TLS**: Always use HTTPS with valid certificates
