# Description: Threatrace Pipeline
# This File is based on the ThreaTrace Paper
# it is a reimplementation of the provided code on github 
# it holds a optimized version of the original code and a very closely related version with the original code
# the original code has just adjustments to run on the new version of pytorch and torch_geometric

# Imports
import os.path as osp
import os
import argparse
import torch
import time
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
import pandas as  pd
from tqdm import tqdm
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import NeighborSampler, DataLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import to_undirected
from typing import Optional, Callable, List, Self, Any, Tuple
import configparser
from torch.optim import Optimizer


#Reproducibility
from torch_geometric import seed_everything


class SAGENet(torch.nn.Module):
    """_summary_
        A two-layer GraphSAGE model 
    Args:
        torch (_type_): _description_
    """
    def __init__(self: Self, in_channels: int, out_channels:int, hidden_layer:int, concat: bool=False):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_layer, normalize=False, concat=concat)
        self.conv2 = SAGEConv(hidden_layer, out_channels, normalize=False, concat=concat)
        #self.lin1 = torch.nn.Linear(256, out_channels)

    def forward(self: Self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        #x = self.lin1(x)
        return F.log_softmax(x, dim=1)


class ThreaTracePipeline():
    # This Class represents a Class, that holds the whole Threatrace Pipeline
    # it contains all the functions and variables needed to run the pipeline
    def __init__(self: Self, config: configparser.ConfigParser, train_data: Data, test_data: Data):
        """_summary_
            Initializes the Threatrace Pipeline Class 
        Args:
            config (_type_): _description_
            train_data (_type_): _description_
            test_data (_type_): _description_
        """
        self.config = config
        self.train_data = train_data
        self.test_data = test_data
        self.train_thre = 1.5
        self.test_thre = 2.0
        self.num_neighbor = -1
        self.shuffle = False
        self.hop = 2
        self.b_train_size = train_data.x.shape[0]
        self.b_test_size = test_data.x.shape[0]
        self.false_classified = []
        self.true_classified = []
        self.init_epochs = 30
        self.submodel_max_epochs = 150
        self.test_cnt_thre = 3
        self.models_dir_path = self.config["Directories"]["models"]
        self.loop_num = 0
        self.hidden_layer = 32
        self.train_model = self.create_model(self.train_data)
        self.test_model = self.create_model(self.test_data)
        self.optimizer = self.create_optimizer(self.train_model, learning_rate=0.001, weight_decay=5e-4)
        self.train_loader = self.create_neighborloader(self.train_data , shuffle=self.shuffle, num_neighbor=self.num_neighbor, hop=self.hop, b_size=self.b_train_size)
        self.test_loader = self.create_neighborloader(self.test_data, shuffle=self.shuffle, num_neighbor=self.num_neighbor, hop=self.hop, b_size=self.b_test_size)
        self.test_data_undirected_edge_index = to_undirected(self.test_data.edge_index, num_nodes=self.test_data.x.shape[0])
        seed_everything(1234)

    def delete_old_models(self: Self):
        """_summary_
            Deletes all old models in the models directory
            This is needed to ensure that the models are not loaded from the last run
        Args:
            self (_type_): _description_
        """
        models = os.listdir(self.models_dir_path)
        for item in models:
            if item.startswith("model_"):
                os.remove(os.path.join(self.models_dir_path, item))
            if item.startswith("tn_"):
                os.remove(os.path.join(self.models_dir_path, item))
            if item.startswith("fp_"):
                os.remove(os.path.join(self.models_dir_path, item))

    def create_model(self: Self, data: Data) -> SAGENet:
        """_summary_
            Creates a new SAGENet Model
        Args:
            self (_type_): _description_
            data (_type_): _description_    
        Returns:
            SAGENet: The created SAGENet Model
        """
        device = torch.device('cpu')
        #train_net = SAGENet
        train_net = SAGENet 
        train_feature_num = data.x.shape[1]
        train_label_num = len(data.y.unique())
        train_model = train_net(train_feature_num, train_label_num, self.hidden_layer).to(device)
        return train_model

    def create_optimizer(self: Self, model: Any, learning_rate: float, weight_decay: float) -> Optimizer:
        """_summary_
            Creates a new Adam Optimizer
        Args:
            self (Self): _description_
            model (Any): _description_
            learning_rate (float): _description_
            weight_decay (float): _description_

        Returns:
            Optimizer: _description_
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        return optimizer

    def create_neighborloader(self, data, shuffle, num_neighbor, hop, b_size, input_nodes = None):
        """_summary_
            Creates a new NeighborLoader for the given data and parameters 
            it is used to create the batches for the training and testing 
            in the Case of GraphSAGE it is used to create the subgraphs and the SAMPLING of the neighbors
        Args:
            data (_type_): _description_
            shuffle (_type_): _description_
            num_neighbor (_type_): _description_
            hop (_type_): _description_
            b_size (_type_): _description_
            input_nodes (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if input_nodes is None:
            input_nodes = data.train_mask
        loader = NeighborLoader(data, shuffle=shuffle, num_neighbors=[num_neighbor] * hop, batch_size=b_size,input_nodes=input_nodes)
        return loader

    def train(self: Self) -> float:
        """_summary_
            Trains the model for one epoch
        Args:
            self (Self): _description_

        Returns:
            float: _description_
        """
        self.train_model.train()
        total_loss = 0
        items = 0
        for data_flow in self.train_loader:
            data_flow.edge_index = add_remaining_self_loops(data_flow.edge_index)[0]
            self.optimizer.zero_grad()
            out = self.train_model(data_flow.x, data_flow.edge_index)
            loss = F.nll_loss(out[:data_flow.batch_size], data_flow.y[:data_flow.batch_size])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * data_flow.batch_size
            items += data_flow.batch_size
        return total_loss / items

    def test(self: Self, model: Any, loader: NeighborLoader) -> float:
        """_summary_
            Tests the model for one epoch
        Args:
            self (Self): _description_
            model (Any): _description_
            loader (NeighborLoader): _description_

        Returns:
            float: _description_
        """
        model.eval()
        correct = 0
        items = 0
        total_loss = 0
        for data_flow in loader:
            data_flow.edge_index = add_remaining_self_loops(data_flow.edge_index)[0]
            out = model(data_flow.x, data_flow.edge_index)
            loss = F.nll_loss(out[:data_flow.batch_size], data_flow.y[:data_flow.batch_size])
            out = out[:data_flow.batch_size]
            pred = out.max(1)[1]
            total_loss += loss.item() * data_flow.batch_size
            correct += pred[:data_flow.batch_size].eq(data_flow.y[:data_flow.batch_size]).sum().item()  
            items += data_flow.batch_size
        return correct / items

    def final_test(self: Self, model: Any, loader: NeighborLoader, data: Data, false_classified: int, true_classified: int) -> Tuple[float, int, int]:
        """_summary_
            Tests the model and returns the false and true classified nodes
            The false and true classified nodes are used to create the new training data for the next submodel
            the classification is done by the train_thre and test_thre parameters
            the thresholds are used to determine the confidence of the decisions
        Args:
            self (Self): _description_
            model (Any): _description_
            loader (NeighborLoader): _description_
            data (Data): _description_
            false_classified (int): _description_
            true_classified (int): _description_

        Returns:
            Tuple[float, int, int]: _description_
        """
        model.eval()
        correct = 0
        items = 0
        data_flow_counter = 0
        for data_flow in loader:
            data_flow_counter += 1
            #print("Data Flow Counter: ", data_flow_counter)
            data_flow.edge_index = add_remaining_self_loops(data_flow.edge_index)[0]
            out = model(data_flow.x, data_flow.edge_index)
            out = out[:data_flow.batch_size]
            #print("Out Shape: ", out.shape)
            bachted_data_flow_n_id = data_flow.n_id[:data_flow.batch_size]
            #print("Batched Data Flow N ID Shape: ", bachted_data_flow_n_id.shape)
            pred = out.max(1)[1]
            #print("Pred Shape: ", pred.shape)
            pro  = F.softmax(out, dim=1)
            #print("Pro Shape: ", pro.shape)
            pro1 = pro.max(1)
            #print(f"Pro1: {pro1}")
            for i in range(len(bachted_data_flow_n_id)):
                pro[i][pro1[1][i]] = -1
            pro2 = pro.max(1)
            #print(f"Pro2: {pro2}")
            for i in range(len(bachted_data_flow_n_id)):
                if pro1[0][i]/pro2[0][i] < self.train_thre:
                    pred[i] = 100
            for i in range(len(bachted_data_flow_n_id)):
                if data.y[bachted_data_flow_n_id[i]] != pred[i]:
                    false_classified.append(int(bachted_data_flow_n_id[i]))
                else:
                    true_classified.append(int(bachted_data_flow_n_id[i]))
            correct += pred[:data_flow.batch_size].eq(data_flow.y[:data_flow.batch_size]).sum().item()  
            items += data_flow.batch_size
            #print(f"Unique Count of data_flow.y:  {data_flow.y.unique(return_counts=True)}")
            #print(f"Unique Count of pred: {pred.unique(return_counts=True)}")
        return correct / items, false_classified, true_classified
    
    def show(self: Self, *s):
        """_summary_
            Prints the given parameters
            *s is used to print multiple parameters
            the printstatement is extended with the current time
        Args:
            self (Self): _description_
        """
        for i in range(len(s)):
            print (str(s[i]) + ' ', end = '')
        print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    def pretraining(self: Self):
        """_summary_
            Trains the first model on all data for init_epochs
        Args:
            self (Self): _description_
        """
            # train first Model on all Data over init_epochs
        for epoch in range(1, self.init_epochs):
            loss = self.train()
            acc = self.test(self.train_model, self.train_loader)
            #tensorboard
            # writer.add_scalars('acc train', {"acc"+str(loop_num): acc}, epoch) # new line
            # writer.add_scalars('loss train', {"loss"+str(loop_num): loss}, epoch)   # new line

            self.show(epoch, loss, acc)
    
    def multi_model_training(self: Self):
        """_summary_
            Trains the submodels on the data
            The submodels are trained until the accuracy is 1 or the loss is below 1, or the submodel_max_epochs is reached
            The submodels represent different "normal" behaviors of the data
            Goal is to create a model for every "normal" behavior of the data
            Problem: if the data is small and not every class is represented we could 
                learn a classifier that says everything this is a NetFlow, or File etc. 
        Args:
            self (Self): _description_
        """
        while (1): # stops if 3 models have 0 true classified, or if acc = 1 
            print("New Round")
            false_classified = []
            true_classified = []
            bad_cnt = 0
            acc, false_classified, true_classified = self.final_test(self.train_model, self.train_loader, self.train_data, false_classified, true_classified)
            #tensorboard:
            
            # writer.add_scalar('false_classified train', len(false_classified), loop_num)   # new line
            # writer.add_scalar('true_classified train', len(true_classified), loop_num)   # new line
            if len(true_classified) == 0:
                bad_cnt += 1
            else: 
                bad_cnt = 0
            if bad_cnt >= self.test_cnt_thre:  # break if 3 times a submodel got 0 true_classified nodes
                break
            print("false_classified, true_classified")
            print(len(false_classified),len(true_classified))
            if len(true_classified) > 0:
                for i in true_classified:
                    self.train_data.train_mask[i] = False
                    self.train_data.test_mask[i] = False
                fw = open(self.models_dir_path + 'fp_feature_label_'+str(self.loop_num)+'.txt', 'w')
                x_list = self.train_data.x[false_classified]
                y_list = self.train_data.y[false_classified]

                if len(x_list) >1:
                    sorted_index = np.argsort(y_list, axis = 0)
                    x_list = np.array(x_list)[sorted_index]
                    y_list = np.array(y_list)[sorted_index]

                for i in range(len(y_list)):
                    fw.write(str(y_list[i])+':')
                    for j in x_list[i]:
                        fw.write(' '+str(j))
                    fw.write('\n')
                fw.close()

                fw = open(self.models_dir_path + 'tn_feature_label_'+str(self.loop_num)+'.txt', 'w')
                x_list = self.train_data.x[true_classified]
                y_list = self.train_data.y[true_classified]

                if len(x_list) >1:
                    sorted_index = np.argsort(y_list, axis = 0)
                    x_list = np.array(x_list)[sorted_index]
                    y_list = np.array(y_list)[sorted_index]

                for i in range(len(y_list)):
                    fw.write(str(y_list[i])+':')
                    for j in x_list[i]:
                        fw.write(' '+str(j))
                    fw.write('\n')
                fw.close()

                torch.save(self.train_model.state_dict(), self.models_dir_path + 'model_'+str(self.loop_num))
                self.show('Model saved loop_num: ' + str(self.loop_num))
                # break #use this break for "single model" setup

            self.train_loader = self.create_neighborloader(self.train_data, shuffle=self.shuffle, num_neighbor=self.num_neighbor, hop=self.hop, b_size=self.b_train_size, input_nodes=self.train_data.train_mask)
            self.train_model = self.create_model(self.train_data)
            self.optimizer = self.create_optimizer(self.train_model, learning_rate=0.01, weight_decay=5e-4)
            self.loop_num  += 1
            acc = 0
            for epoch in range(1, self.submodel_max_epochs):
                loss = self.train()
                acc = self.test(self.train_model, self.train_loader)
                #tensorboard
                # writer.add_scalars('acc train', {"acc"+str(loop_num): acc}, epoch) # new line
                # writer.add_scalars('loss train', {"loss"+str(loop_num): loss}, epoch)   # new line
                #show(epoch, loss, acc)
                if loss < 1: break
            if acc == 1: break 
    show('Finish training graph')

    def test_model_performance(self: Self, max_runs: int=100):
        # todo: this loop has to be optimized, it still based on the original code
        """_summary_
            Tests the model performance on the test data
            Iterates over all models and tests them on the test data
            The models are loaded from the models directory
            The Results are saved in the test_data.test_mask 
            The test_data.test_mask holds a boolean for every node in the test_data
            If the boolean is true the node is classified as normal
        Args:
            self (Self): _description_
            max_runs (int, optional): _description_. Defaults to 100.
        """
            #runime 8min
        loop_num = 0
        
        while(1):
            if loop_num > max_runs: break
            model_path = self.models_dir_path+ 'model_'+str(loop_num)
            if not osp.exists(model_path): 
                loop_num += 1
                continue
            self.test_model.load_state_dict(torch.load(model_path))
            false_classified = []	
            true_classified = [] 
            test_acc, false_classified, true_classified = self.final_test(self.test_model, self.test_loader, self.test_data, false_classified, true_classified)
            #writer.add_scalar('acc test', acc, loop_num)   # new line
            print("Loop_num: " + str(loop_num) + '  Accuracy:{:.4f}'.format(test_acc) + '  true_classified:' + str(len(true_classified))+ '  false_classified:' + str(len(false_classified)))
            for i in true_classified:
                self.test_data.test_mask[i] = False
            if test_acc == 1: break
            loop_num += 1
        print(f"Unique Count of data_flow.y:  {self.test_data.y.unique(return_counts=True)}")
        print(f"Unique Count of pred: {self.test_data.test_mask.unique(return_counts=True)}")
            # 51 vs 57 features problems

    def get_detection_insights(self: Self) -> torch.Tensor:
        """
        _summary_
            Returns the detection insights
            Unique Count of data_flow.y / Unique Count of Prediction (test_data.test_mask)
        Args:
            self (Self): _description_
        Returns:
            torch.Tensor: _description_
        """
        print("Get Detection Insights")
        print(f"Unique Count of data_flow.y:  {self.test_data.y.unique(return_counts=True)}")
        print(f"Unique Count of pred: {self.test_data.test_mask.unique(return_counts=True)}")
        filtered_tensor = self.test_data.y[self.test_data.test_mask]
        print(f"Unique Count of filtered_tensor: {filtered_tensor.unique(return_counts=True)}")
        return filtered_tensor

    def evaluation(self: Self, gt: List[int]):
        """_summary_
            Evaluates the model performance on the test data
            The evaluation is done by comparing the test_data.test_mask with the ground truth
            The ground truth is given by the gt parameter
            The gt parameter is a list of all nodes that are malicious
            The test_data.test_mask holds a boolean for every node in the test_data
            If the boolean is true the node is classified as normal
            The Nodes are evaluated as follows:
                True Positive: The Node is classified as normal and is normal
                False Positive: The Node is classified as normal but is malicious
                True Negative: The Node is classified as malicious and is malicious
                False Negative: The Node is classified as malicious but is normal
            To detemine the FPs/TNs/FNs/TPs the following steps are done:
                1. Every Node is classified as TN
                2. Every Node in the gt is classified as FN
                3. Every Node in the test_data.test_mask is visited and the neighbors are checked
                4. If the node is in the gt it is classified as TP
                5. If the node is in the neighbors of a node in the gt it is classified as TP
                6. If the node is not the neighbors of a node in the gt it is classified as FP
        Args:
            self (Self): _description_
            gt (_type_): _description_

        Returns:
            _type_: _description_
        """
        flag =0
        eps = 1e-10
        eval_len = len(self.test_data.x)
        ans = np.full(eval_len, 'tn') # first: every node is a True Negative 
        unique_values, counts = np.unique(ans, return_counts=True)
        print(dict(zip(unique_values, counts)))
        print("start")
        print(dict(zip(unique_values, counts)))
        #breakpoint()
        print("set every node in gt to fn")

        #ans = ['fn' if idx < eval_len else continue for idx in gt]
        for idx in gt:
            if idx < eval_len:
                ans[idx] = 'fn'
        #ans = ['fn' for idx in gt if idx < eval_len]
        unique_values, counts = np.unique(ans, return_counts=True)
        print(dict(zip(unique_values, counts)))
        #ans[gt] = 'fn' # second: every node in the ground truth is a False Negative
        print("gt")
        unique_values, counts = np.unique(ans, return_counts=True)
        print(dict(zip(unique_values, counts)))

        hits = self.test_data.test_mask
        hit_indexes = torch.nonzero(hits).squeeze().tolist()
        intersection_counter = 0 
        for element in tqdm(hit_indexes, total=len(hit_indexes)):
            if element:
                i_as_list = [element]
                hit_neighbors, edge_index, mapping, edge_mask = k_hop_subgraph(i_as_list, num_hops=2, edge_index=self.test_data_undirected_edge_index) # get neighbors of hit node
                intersection = list(set(gt) & set(hit_neighbors.tolist())) #-> performanze boost! 
                if intersection != []:
                    intersection_counter += 1
                    for index in intersection:
                        if index < len(ans):
                            ans[index] = 'tp'
                        else:
                            print("index out of bounds")
                            print(index)
                    flag = 1 
                if element in gt: # if hit node is in ground truth (direct hit)
                    if element < len(ans):
                        ans[element] = 'tp'
                    else:
                        print("index out of bounds")
                        print(element)
                    flag = 1
                else:
                    if flag == 0: # wenn kein direkter oder indirekter hit
                        if element < len(ans):
                            ans[element] = 'fp'
                        else:
                            print("index out of bounds")
                            print(element)
                        
        unique_values, counts = np.unique(ans, return_counts=True)
        print(dict(zip(unique_values, counts)))
        print(f"Intersection Counter: {intersection_counter}")


        tn = 0
        tp = 0
        fn = 0
        fp = 0
        count = 0
        for i in ans:
            if i == 'tp': tp += 1
            if i == 'tn': tn += 1
            if i == 'fp': 
                fp += 1
            if i == 'fn': 
                fn += 1
                #f_fn.write(str(count) + "\n")
            count = count +1
        print(count)
        print(tp,fp,tn,fn)
        print("TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}".format(tp = tp, fp = fp, tn = tn, fn = fn))
        precision = tp/(tp+fp+eps)
        recall = tp/(tp+fn+eps)
        fscore = 2*precision*recall/(precision+recall+eps)
        # writer.add_scalar('Precision', precision, 0)   # new line
        # writer.add_scalar('Recall', recall,0)   # new line
        # writer.add_scalar('F-Score', fscore,0)   # new line
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('F-Score: ', fscore)
        print("Accuracy: ", accuracy)

        return hit_indexes
        # geschönt und fertig :D 
        #TP: 8389, FP: 0, TN: 513082, FN: 52867
        #TP: 2177, FP: 1, TN: 549882, FN: 22278
        #TP: 16, FP: 3710, TN: 557771, FN: 12841
        #TP: 4210, FP: 1, TN: 549882, FN: 20245

        #TP: 4568, FP: 0, TN: 549883, FN: 19887


    def evaluation_single_hop(self: Self, gt: List[int]):
        """_summary_
            Evaluates the model performance on the test data
            Same as evaluation but only one hop is used
        Args:
            self (Self): _description_
            gt (List[int]): _description_

        Returns:
            _type_: _description_
        """
        flag =0
        eps = 1e-10
        eval_len = len(self.test_data.x)
        ans = np.full(eval_len, 'tn') # first: every node is a True Negative 
        unique_values, counts = np.unique(ans, return_counts=True)
        print("start")
        print(dict(zip(unique_values, counts)))
        #breakpoint()
        print("set every node in gt to fn")
        for idx in gt:
            if idx < eval_len:
                ans[idx] = 'fn'
        #ans[gt] = 'fn' # second: every node in the ground truth is a False Negative        print("gt")
        unique_values, counts = np.unique(ans, return_counts=True)
        print(dict(zip(unique_values, counts)))

        hits = self.test_data.test_mask
        hit_indexes = torch.nonzero(hits).squeeze().tolist()
        for index, element in tqdm(enumerate(hit_indexes), total=len(hit_indexes)):
            if element:
                # i_as_list = [index]
                # hit_neighbors, edge_index, mapping, edge_mask = k_hop_subgraph(i_as_list, num_hops=2, edge_index=self.test_data.edge_index, flow="target_to_source") # get neighbors of hit node
                # intersection = list(set(gt) & set(hit_neighbors.tolist())) #-> performanze boost! 
                # if intersection != []:
                #     ans[intersection] = 'tp'
                #     flag = 1 
                if element in gt: # if hit node is in ground truth (direct hit)
                    if index < len(ans):
                        ans[element] = 'tp'
                    else:
                        print("index out of bounds")
                        print(index)
                    flag = 1
                else:
                    # if flag == 0: # wenn kein direkter oder indirekter hit
                    #     ans[element] = 'fp'
                    if flag == 0: # wenn kein direkter oder indirekter hit
                        if element < len(ans):
                            ans[element] = 'fp'
                        else:
                            print("index out of bounds")
                            print(element)
        unique_values, counts = np.unique(ans, return_counts=True)
        print(dict(zip(unique_values, counts)))


        tn = 0
        tp = 0
        fn = 0
        fp = 0
        count = 0
        for i in ans:
            if i == 'tp': tp += 1
            if i == 'tn': tn += 1
            if i == 'fp': 
                fp += 1
            if i == 'fn': 
                fn += 1
                #f_fn.write(str(count) + "\n")
            count = count +1
        print(count)
        print(tp,fp,tn,fn)
        print("TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}".format(tp = tp, fp = fp, tn = tn, fn = fn))
        precision = tp/(tp+fp+eps)
        recall = tp/(tp+fn+eps)
        fscore = 2*precision*recall/(precision+recall+eps)
        # writer.add_scalar('Precision', precision, 0)   # new line
        # writer.add_scalar('Recall', recall,0)   # new line
        # writer.add_scalar('F-Score', fscore,0)   # new line
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('F-Score: ', fscore)
        print("Accuracy: ", accuracy)
        return hit_indexes

    def reinit_test_data(self):
         self.test_data.test_mask = torch.tensor([True] * len(self.test_data.y), dtype=torch.bool)
       
         self.test_loader = self.create_neighborloader(self.test_data, shuffle=self.shuffle, num_neighbor=self.num_neighbor, hop=self.hop, b_size=self.b_test_size, input_nodes=self.test_data.test_mask)



############################################################################################################
#### ThreaTrace Original Code adapted for new Version of PyTorch Geometric
############################################################################################################
#### Orig Data Preprocessing
### cadets_train.txt and cadets_test.txt are the original data files
### they are created with the script "parse_darpatc.py"
training_data_path = "/Users/robinbuchta/Documents/GitHub/threaTrace/graphchi-cpp-master/graph_data/darpatc/cadets_train.txt"
test_data_path = "/Users/robinbuchta/Documents/GitHub/threaTrace/graphchi-cpp-master/graph_data/darpatc/cadets_test.txt"

class ThreaTraceCadets(InMemoryDataset):
    def __init__(self, df):
        super().__init__('.')
        node_cnt = 0
        nodeType_cnt = 0
        edgeType_cnt = 0
        provenance = []
        nodeType_map = {}
        edgeType_map = {}
        edge_s = []
        edge_e = []
        data_thre = 1000000
        #cnt = 0
        for out_loop in range(1):
            f = open(training_data_path, 'r')
            nodeId_map = {}
            for line in f:
                #cnt += 1
                #if cnt == 1: continue #psacal 
                temp = line.strip('\n').split('\t')
                #temp = line.strip('\n').split(',')
                if not (temp[0] in nodeId_map.keys()):
                    nodeId_map[temp[0]] = node_cnt
                    node_cnt += 1
                temp[0] = nodeId_map[temp[0]]	
                if not (temp[2] in nodeId_map.keys()):
                    nodeId_map[temp[2]] = node_cnt
                    node_cnt += 1
                temp[2] = nodeId_map[temp[2]]
                if not (temp[1] in nodeType_map.keys()):
                    nodeType_map[temp[1]] = nodeType_cnt
                    nodeType_cnt += 1
                temp[1] = nodeType_map[temp[1]]
                if not (temp[3] in nodeType_map.keys()):
                    nodeType_map[temp[3]] = nodeType_cnt
                    nodeType_cnt += 1
                temp[3] = nodeType_map[temp[3]]
                if not (temp[4] in edgeType_map.keys()):
                    edgeType_map[temp[4]] = edgeType_cnt
                    edgeType_cnt += 1
                temp[4] = edgeType_map[temp[4]]
                edge_s.append(temp[0])
                edge_e.append(temp[2])
                provenance.append(temp)
        f_train_feature = open(feature_path, 'w')
        for i in edgeType_map.keys():
            f_train_feature.write(str(i)+'\t'+str(edgeType_map[i])+'\n')
        f_train_feature.close()
        f_train_label = open(label_path, 'w')
        for i in nodeType_map.keys():
            f_train_label.write(str(i)+'\t'+str(nodeType_map[i])+'\n')
        f_train_label.close()
        feature_num = edgeType_cnt
        label_num = nodeType_cnt
        x_list = []
        y_list = []
        train_mask = []
        test_mask = []
        for i in range(node_cnt):
            temp_list = []
            for j in range(feature_num*2):
                temp_list.append(0)
            x_list.append(temp_list)
            y_list.append(0)
            train_mask.append(True)
            test_mask.append(True)
        for temp in provenance:
            srcId = temp[0]
            srcType = temp[1]
            dstId = temp[2]
            dstType = temp[3]
            edge = temp[4]
            x_list[srcId][edge] += 1
            #x_list[srcId][edge] = 1 # modify feature on boolean instead of count
            y_list[srcId] = srcType
            x_list[dstId][edge+feature_num] += 1
            #x_list[dstId][edge+feature_num] = 1 # modify feature on boolean instead of count
            y_list[dstId] = dstType
        x = torch.tensor(x_list, dtype=torch.float)	
        y = torch.tensor(y_list, dtype=torch.long)
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        test_mask = train_mask
        edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
        self.data, self.slices = self.collate([data])
        @property
        def processed_file_names(self):
            return './cadets_data.pt'

def create_tt_orig_data_train():
    train_dataset = ThreaTraceCadets(training_data_path)
    train_data = train_dataset[0]
    train_data.n_id = torch.arange(train_data.num_nodes) # Assign each node its global node index
    return train_data


class ThreaTraceCadetsTest(InMemoryDataset):
	def __init__(self, data_list):
		super(ThreaTraceCadetsTest, self).__init__('/tmp/TestDataset')
		self.data, self.slices = self.collate(data_list)

	def _download(self):
		pass
	def _process(self):
		pass

models_dir_path = "/Users/robinbuchta/Documents/GitHub/threaTrace/models/"
feature_path = models_dir_path+ "feature.txt"
label_path = models_dir_path + "/label.txt"
ground_truth_path = "/Users/robinbuchta/Documents/GitHub/threaTrace/groundtruth/groundtruth_nodeId_modified.txt" #without Files
id_uuid_map_path = "/Users/robinbuchta/Documents/GitHub/threaTrace/groundtruth/id_to_uuid.txt"
ground_truth_uuid_list = "/Users/robinbuchta/Documents/GitHub/threaTrace/groundtruth/cadets.txt"
alarm_path = "/Users/robinbuchta/Documents/GitHub/threaTrace/groundtruth/alarm.txt"
def MyDataset(path):
		feature_num = 0
		label_num = 0
		f_feature = open(feature_path, 'r')
		feature_map = {}
		for i in f_feature:
			temp = i.strip('\n').split('\t')
			feature_map[temp[0]] = int(temp[1])
			feature_num += 1
		f_feature.close()

		f_label = open(label_path, 'r')
		label_map = {} 
		for i in f_label:
			temp = i.strip('\n').split('\t')
			label_map[temp[0]] = int(temp[1])
			label_num += 1
		f_label.close()

		#groundtruth_uuid.txt
		f_gt = open(ground_truth_uuid_list, 'r')
		ground_truth = {}
		for line in f_gt:
			ground_truth[line.strip('\n')] = 1
		f_gt.close()
		node_cnt = 0
		nodeType_cnt = 0
		edgeType_cnt = 0
		provenance = []
		nodeType_map = {}
		edgeType_map = {}
		edge_s = []
		edge_e = []
		adj = {}
		adj2 = {}
		data_thre = 1000000
		fw = open(ground_truth_path, 'w+')
		fw2 = open(id_uuid_map_path, 'w+')
		nodeId_map = {}
		cnt = 0
		for i in range(1):
			now_path = path
			#show(now_path)
			f = open(now_path, 'r')
			for line in f:
				cnt += 1
				#if cnt == 1: continue # für pascals datensatz
				#print(line)
				if cnt % 1000000 == 0: nothing = 1  #show(str(cnt))
				temp = line.strip('\n').split('\t')
				#temp = line.strip('\n').split(',')
				if not (temp[1] in label_map.keys()): continue
				if not (temp[3] in label_map.keys()): continue
				if not (temp[4] in feature_map.keys()): continue

				if not (temp[0] in nodeId_map.keys()):
					nodeId_map[temp[0]] = node_cnt
					fw2.write(str(node_cnt) + ' ' + temp[0] + '\n')

					if temp[0] in ground_truth.keys():
						fw.write(str(nodeId_map[temp[0]])+' '+temp[1]+' '+temp[0]+'\n')
					node_cnt += 1

				temp[0] = nodeId_map[temp[0]]	

				if not (temp[2] in nodeId_map.keys()):
					nodeId_map[temp[2]] = node_cnt
					fw2.write(str(node_cnt) + ' ' + temp[2] + '\n')

					if temp[2] in ground_truth.keys():
						fw.write(str(nodeId_map[temp[2]])+' '+temp[3]+' '+temp[2]+'\n')
					node_cnt += 1
				temp[2] = nodeId_map[temp[2]]		
				temp[1] = label_map[temp[1]]
				temp[3] = label_map[temp[3]]
				temp[4] = feature_map[temp[4]]
				edge_s.append(temp[0])
				edge_e.append(temp[2])
				if temp[2] in adj.keys():
					adj[temp[2]].append(temp[0])
				else:
					adj[temp[2]] = [temp[0]]
				if temp[0] in adj2.keys():
					adj2[temp[0]].append(temp[2])
				else:
					adj2[temp[0]] = [temp[2]]
				provenance.append(temp)
			f.close()
		fw.close()
		fw2.close()
		x_list = []
		y_list = []
		train_mask = []
		test_mask = []
		for i in range(node_cnt):
			temp_list = []
			for j in range(feature_num*2):
				temp_list.append(0)
			x_list.append(temp_list)
			y_list.append(0)
			train_mask.append(True)
			test_mask.append(True)

		for temp in provenance:
			srcId = temp[0]
			srcType = temp[1]
			dstId = temp[2]
			dstType = temp[3]
			edge = temp[4]
			x_list[srcId][edge] += 1
			#x_list[srcId][edge] = 1 # modify feature on boolean instead of count
			y_list[srcId] = srcType
			x_list[dstId][edge+feature_num] += 1
			#x_list[dstId][edge+feature_num] = 1 # modify feature on boolean instead of count
			y_list[dstId] = dstType

		x = torch.tensor(x_list, dtype=torch.float)	
		y = torch.tensor(y_list, dtype=torch.long)
		train_mask = torch.tensor(train_mask, dtype=torch.bool)
		test_mask = torch.tensor(test_mask, dtype=torch.bool)
		edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
		#data1 = Data(x=x, y=y,edge_index=edge_index, train_mask=train_mask, test_mask = test_mask)
		feature_num *= 2
		data1 = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
		#self.data, self.slices = self.collate([data])
		
		return [data1], adj, adj2

def create_tt_orig_data_test():
    tmp_data, adj, adj2 = MyDataset(test_data_path) 
    test_dataset = ThreaTraceCadetsTest(tmp_data)
    test_data = test_dataset[0]
    test_data.n_id = torch.arange(test_data.num_nodes)
    return test_data, adj, adj2

def original_evaluation(adj, adj2, test_data):

    #fw = open('/home/buchta/threaTrace/threaTrace/models/tmp/alarm.txt', 'w+')
    fw =open(alarm_path, 'w+')
    fw.write(str(len(test_data.test_mask))+'\n')

    print(f"len test_data.test_mask: {len(test_data.test_mask)}")
    print(f"Unique value count of testmask: {np.unique(test_data.test_mask, return_counts=True)}")
    for i in tqdm(range(len(test_data.test_mask))):
        if test_data.test_mask[i] == True: #Unzureichend klassifizierte Knoten 
            fw.write('\n')
            fw.write(str(i)+':')
            ##Evaluation on Neighborhood
            neibor = set()
            if i in adj.keys():
                for j in adj[i]:
                    neibor.add(j)
                    if not j in adj.keys(): continue
                    for k in adj[j]:
                        neibor.add(k)  # up to 2 hop Neighborhood source
            if i in adj2.keys():
                for j in adj2[i]:
                    neibor.add(j)
                    if not j in adj2.keys(): continue
                    for k in adj2[j]:
                        neibor.add(k) # up to 2 hop Neighborhood destination 
            neibor = set(neibor)
            for j in neibor:
                fw.write(' '+str(j))
    #fw.close()

    print('Finish testing graph')

    f_gt = open(ground_truth_path, 'r')
    f_alarm = open(alarm_path, 'r')
    eps = 1e-10

    gt = {}
    for line in f_gt:
        gt[int(line.strip('\n').split(' ')[0])] = 1
    ans = []

    for line in f_alarm: # run over all alarms (wrongly classified nodes )
        if line == '\n': continue
        if not ':' in line: # only first wirting = len of nodes 
            print("no : in line")
            print(line)
            tot_node = int(line.strip('\n'))
            for i in range(tot_node):
                ans.append('tn')  # Placeholder, every node is True Negativ
            for i in gt:
                ans[i] = 'fn'     # Every Node which is in Ground_Truth is in first Place False Negative
            continue
        # Now fill List with real values 
        line = line.strip('\n')
        a = int(line.split(':')[0]) #wrongly classified node 
        b = line.split(':')[1].strip(' ').split(' ') # neighborhood of wrongly classified node 
        flag = 0
        for i in b:
            if i == '': continue
            if int(i) in gt.keys(): #indirecte Treffer über die Nachbarschaft 
                ans[int(i)] = 'tp'
                flag = 1   #wenn nachbarschaft auch merkwürdig ist dann flag = 1 
        if a in gt.keys(): #direkte Treffer auf knoten direkt 
            ans[a] = 'tp'
        else: # orig code
            if flag == 0: # wenn nachbarschaft nicht merkwürdig ist dann ist es ein Falsch positiver 
                ans[a] = 'fp'


    tn = 0
    tp = 0
    fn = 0
    fp = 0
    count = 0
    for i in ans:
        if i == 'tp': tp += 1
        if i == 'tn': tn += 1
        if i == 'fp': 
            fp += 1
        if i == 'fn': 
            fn += 1
        count = count +1
    print(count)
    print(tp,fp,tn,fn)
    print("TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}".format(tp = tp, fp = fp, tn = tn, fn = fn))
    precision = tp/(tp+fp+eps)
    recall = tp/(tp+fn+eps)
    fscore = 2*precision*recall/(precision+recall+eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F-Score: ', fscore)
    print("Accuracy: ", accuracy)
    #TP: 12851, FP: 21394, TN: 322928, FN: 1 # with single model 
