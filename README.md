# gnn_testenv
Graph Neural Network Test Environment for Anomaly Detection in Provenance Graph Data 

## 1. Install Docker 
https://docs.docker.com/engine/install/ 

## 2. Start Docker Compose 
First change the path in docker-compose.yml of the volume to the path were your want to store the data e.g.  <here own path>:/output/ -> /home/buchta/gnn_testenv/DARPA/:/output/



```bash
cd db 
docker compose up -d
cd .. 
```

Adminder is available at http://localhost:8080
The database is available at localhost:5432

## 3. Install Pytorch Geometric
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html 

## 4. Install the requirements.txt
```bash
pip install -r requirements.txt
```
## 5. Install optional packages for PyG based on your CUDA version
e.g.
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```
## 5. Install package 
```bash
pip install -e .
```
## 6. Place the data in the data folder
```bash
cp ta1-cadets-e3-official-2.json.tar.gz /home/buchta/gnn_testenv/DARPA/cadets/03_record 
```
## 7. Replace the paths in the config file
```bash
configs/eng3/cadets/cadets_03_record.ini
```
## 8. Run the Example script with Juypter Notebook 