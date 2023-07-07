# gnn_testenv
Graph Neural Network Test Environment for Anomaly Detection in Provenance Graph Data 

# 1. Install Docker 
https://docs.docker.com/engine/install/ 

# 2. Start Docker Compose 

```bash
cd db 
docker compose up -d
cd .. 
```

Adminder is available at http://localhost:8080
The database is available at localhost:5432

# 3. Install Pytorch Geometric
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html 

# 4. Install the package
```bash
pip install -e .
```