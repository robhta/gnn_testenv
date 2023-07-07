# This file is used to construct the graph for the Threatrace dataset
# The graph is constructed using the PyTorch Geometric library

# Import the required libraries
import logging
from benchmark.db_import import * 
from benchmark.utils import *
from benchmark.ground_truth import *
import pandas as pd
import torch
from typing import List, Tuple, Dict, Union, Callable, Optional, Type
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import configparser


#############################
# Create Edge List
def create_threatrace_edge_list_for_both(config: configparser.ConfigParser, own_timestamp_percent: int = 0) -> None:  
        """_summary_
        create_threatrace_edge_list_for_both train and test
        Args:
            config (configparser.ConfigParser): _description_
            own_timestamp_percent (int, optional): _description_. Defaults to 0.
        """
        logging.info(f"create_threatrace_edge_list_for_both")
        create_threatrace_node_type(config)
        join_information_for_threatrace(config) 
       # create_threatrace_benign_edge_list(config)
        drop_test_edges(config, own_timestamp_percent)

# possible optimization, filter after recieving the data from the database
def create_threatrace_edge_list(config: configparser.ConfigParser, test_bool: bool=False):
        """_summary_
                create the edge list for the PyG Data Object
        Args:
            config (configparser.ConfigParser): _description_
            test_bool (bool, optional): _description_. Defaults to False.
        """
        logging.info(f"create_threatrace_edge_list")
        if test_bool:
                threatrace_edges = config['threatrace']['test_edges']
        else:
                threatrace_edges = config['threatrace']['train_edges'] 
#         create_threatrace_node_type(config)
#         join_information_for_threatrace(config) 
#        # create_threatrace_benign_edge_list(config)
#         drop_test_edges(config)
        rearrange_nodes(config, test_bool)
        query1 = f"CREATE INDEX IF NOT EXISTS src_num ON {threatrace_edges} (src_num);"
        query2 = f"CREATE INDEX IF NOT EXISTS dst_num ON {threatrace_edges} (dst_num);"
        query3 = f"CREATE INDEX IF NOT EXISTS event_type ON {threatrace_edges} (event_type);"
        add_node_label_mapping(config, test_bool)
        execute_query(config, query1)
        execute_query(config, query2)
        execute_query(config, query3)

def create_threatrace_node_type(config: configparser.ConfigParser) -> None:
        """_summary_
                create the node type for the PyG Data Object
                it is saved as a new column in the nodes table
        Args:
            config (configparser.ConfigParser): _description_
        """
        logging.info(f"create_threatrace_node_type:")
        table_nodes = config['SQL']['table_name_nodes']
        # create two new columns: subtype and whole_type 
        subtypes = "subtypes"
        whole_type = "whole_type"
        query = f'ALTER TABLE {table_nodes} ADD {subtypes} VARCHAR(255) NULL;'
        query1 = f'ALTER TABLE {table_nodes} ADD {whole_type} VARCHAR(255) NULL;'
        # update subtype and whole_type with the Ideas of ThreaTrace
        concat_cols = "subject_type, srcsinkobject_type, fileobject_type, principal_type"
        #where_clause = "node_type != 'UnnamedPipeObject' AND node_type != 'NetFlowObject' AND node_type != 'MemoryObject'"
        query2 = f"UPDATE {table_nodes} SET {subtypes} = CONCAT({concat_cols});" #WHERE {where_clause};"
        concat_cols_string = "node_type, '_', subtypes"
        where_clause2 = "node_type != 'Host' AND node_type != 'Timemarker' AND node_type != 'Endmarker' AND node_type != 'Unitdependency'"
        query3 = f"UPDATE {table_nodes} SET {whole_type} = CONCAT({concat_cols_string}) WHERE {where_clause2};"

        execute_query(config, query)
        execute_query(config, query1)
        execute_query_and_commit(config, query2)
        execute_query_and_commit(config, query3)
        logging.info(f"create_threatrace_node_type: done")

def join_information_for_threatrace(config: configparser.ConfigParser) -> None:
        """_summary_
                join the information for the Threatrace dataset
        Args:
                config (configparser.ConfigParser): _description_
        """
        #runtime 2m 40s
        table_edges = config['SQL']['table_name_edges']
        table_nodes = config['SQL']['table_name_nodes']
        tmp1 = "tmp1"
        tmp2 = "tmp2"
        whole_type = "whole_type"
        query = f"""CREATE OR REPLACE TABLE {tmp1} AS       
                SELECT e.id
                        , e.event_subject_com_bbn_tc_schema_avro_cdm18_uuid as src_uuid
                        , e.event_predicateobject_com_bbn_tc_schema_avro_cdm18_uuid as dst_uuid
                        , e.event_predicateobject2_com_bbn_tc_schema_avro_cdm18_uuid as dst2_uuid
                        , e.event_type
                        , e.event_timestampnanos
                        , e.source_numeration as src_num
                        , e.destination_numeration as dst_num
                        , e.destination2_numeration as dst2_num
                        , n1.{whole_type} as type_s 
                        , n2.{whole_type} as type_d1
                        , n3.{whole_type} as type_d2
                FROM {table_edges} e  
                        LEFT OUTER JOIN {table_nodes} n1 ON n1.uuid_numeration = e.source_numeration
                        LEFT OUTER JOIN {table_nodes} n2 ON n2.uuid_numeration = e.destination_numeration
                        LEFT OUTER JOIN {table_nodes} n3 ON n3.uuid_numeration = e.destination2_numeration
                        WHERE (e.source_numeration IS NOT NULL AND e.destination_numeration IS NOT NULL)     
                        OR (e.source_numeration IS NOT NULL AND e.destination2_numeration IS NOT NULL)            
                ;"""
        query2 = f""" CREATE OR REPLACE TABLE {tmp2}
                SELECT id
                        , src_uuid
                        , src_num
                        , type_s
                        , dst_uuid 
                        , dst_num 
                        , type_d1 as dst_type
                        , event_type
                        , event_timestampnanos
                FROM {tmp1}
                UNION ALL
                SELECT id
                        , src_uuid
                        , src_num
                        , type_s
                        , dst2_uuid as dst_uuid
                        , dst2_num as dst_num
                        , type_d2 as dst_type
                        , event_type
                        , event_timestampnanos
                FROM {tmp1} WHERE dst2_num IS NOT NULL        
                """
        #query3 = f"""DROP TABLE {tmp1};"""
        execute_query(config, query)
        execute_query(config, query2)
        #execute_query(config, query3)

def drop_test_edges(config: configparser.ConfigParser, own_timestamp_percent:int =0) -> None:
        """_summary_
                drop the test edges from the edge list
        Args:
                config (configparser.ConfigParser): _description_
                own_timestamp_percent (int, optional): _description_. Defaults to 0.
        """
        threatrace_edges_benign = "tmp2_benign"
        tmp2 = "tmp2"
        time_column = "event_timestampnanos"
        yaml_data = read_gt_yaml(config)
        date, start_ts, end_ts = extract_time_information(yaml_data)
        # Define the time range
        start_time_unix_ns = parse_darpa_gt_to_unix_ns(date=date, time=start_ts, plus=False)
        end_time_unix_ns = parse_darpa_gt_to_unix_ns(date=date, time=end_ts, plus=True)
        if own_timestamp_percent != 0:
                query_ts = f"""SELECT MIN({time_column}) AS min_value, MAX({time_column}) AS max_value FROM {tmp2};"""
                results_query_ts = execute_query(config, query_ts)
                min_value, max_value = results_query_ts[0]
                percentage = own_timestamp_percent / 100  # 0,1%
                range_length = max_value - min_value
                start_time_unix_ns = min_value + (range_length * percentage) # timestamp from above isnt used here, its a new one (just same name for the variable)
        # time_range_condition = f"{time_column} BETWEEN '{start_time_unix_ns}' AND '{end_time_unix_ns}'"
        
        # query = f"""CREATE OR REPLACE TABLE {threatrace_edges_benign} AS
        #         SELECT *
        #         FROM {tmp2}
        #         WHERE {time_column} BETWEEN '{start_time_unix_ns}' AND '{end_time_unix_ns}';"""
        query = f"""CREATE OR REPLACE TABLE {threatrace_edges_benign} AS
                SELECT *
                FROM {tmp2}
                WHERE {time_column} < '{start_time_unix_ns}';"""
        execute_query(config, query)


def rearrange_nodes(config: configparser.ConfigParser, test_bool:bool=False):
        """_summary_
                rearrange the nodes in the nodes table
                it is for test and train, based on the bool
        Args:
                config (configparser.ConfigParser): _description_
                test_bool (bool, optional): _description_. Defaults to False.
        """
        logging.info("rearrange_nodes")
        if test_bool:
                edges = config['threatrace']['test_edges']
                nodes = config['threatrace']['test_nodes']
                tmp2 = "tmp2"
        else:
                edges = config['threatrace']['train_edges']
                nodes = config['threatrace']['train_nodes']
                tmp2 = "tmp2_benign"
        query = f"""
                CREATE OR REPLACE TABLE {nodes} AS
                SELECT DISTINCT src_num AS num, src_uuid AS node, type_s as node_type FROM {tmp2}
                UNION
                SELECT DISTINCT dst_num AS num, dst_uuid AS node, dst_type as node_type FROM {tmp2} 
                ORDER BY num;"""
        query2 = f"""
                ALTER TABLE {nodes} ADD id INTEGER UNIQUE KEY AUTO_INCREMENT;"""
        query3 = f"""
                CREATE INDEX IF NOT EXISTS node_id ON {nodes} (id);
                """
        query4 = f"""
                CREATE INDEX IF NOT EXISTS num ON {nodes} (num);
                """
        query5 = f""" 
                CREATE OR REPLACE TABLE {edges} AS
                SELECT e.id
                        , src_uuid
                        , src_num
                        , type_s
                        , dst_uuid 
                        , dst_num 
                        , dst_type
                        , event_type
                        , event_timestampnanos
                        , n1.id as src_num_new
                        , n2.id as dst_num_new
                FROM {tmp2} e
                JOIN {nodes} n1 ON e.src_num=n1.num 
                JOIN {nodes} n2 ON e.dst_num=n2.num
        ;"""
        query6 = f""" DROP TABLE {tmp2};"""

        execute_query(config, query)
        execute_query(config, query2)
        execute_query(config, query3)
        execute_query(config, query4)
        execute_query(config, query5)
        execute_query(config, query6)

def add_node_label_mapping(config: configparser.ConfigParser, test_bool: bool=False) -> None:
        """_summary_
                add the node label mapping to the nodes table
                it is for test and train, based on the bool     
        Args:
                config (configparser.ConfigParser): _description_
                test_bool (bool, optional): _description_. Defaults to False.
        """     
        if test_bool:
                nodes = config['threatrace']['test_nodes']
        else:
                nodes = config['threatrace']['train_nodes']
        original_column = "node_type"
        mapped_column = "node_type_mapped"
        # Execute the SQL query to fetch distinct values of the original column
        query7 = f"SELECT DISTINCT {original_column} FROM {nodes}"
        distinct_values = execute_query(config, query7)

        # Create a mapping dictionary
        mapping = {}

        # Generate the mapping from distinct values to integers
        for i, value in enumerate(distinct_values):
                mapping[value[0]] = i

        # Alter the table to add a new column for the mapped values
        alter_query = f"ALTER TABLE {nodes} ADD COLUMN {mapped_column} INT"
        execute_query_and_commit(config, alter_query)

        # Update the table with the mapped values
        update_query = f"UPDATE {nodes} SET {mapped_column} = CASE "
        for value, mapped_value in mapping.items():
                update_query += f"WHEN {original_column} = '{value}' THEN {mapped_value} "
        update_query += "ELSE NULL END"
        execute_query_and_commit(config, update_query)

# def add_node_features(config):
#         nodes_table = "threatrace_nodes"
#         edges_table = "threatrace_edges"
#         edge_type = "event_type"
#         # Execute the SQL query to fetch distinct values of the original column
#         query7 = f"SELECT DISTINCT {edge_type} FROM {edges_table}"
#         distinct_values = execute_query(config, query7)

#         # Alter the table to add new columns based on the distinct values
#         for value in distinct_values:
#                 value = value[0]
#                 alter_query_in = f"ALTER TABLE {nodes_table} ADD COLUMN {value}_in INT"
#                 alter_query_out = f"ALTER TABLE {nodes_table} ADD COLUMN {value}_out INT"
#                 execute_query_and_commit(config, alter_query_in)
#                 execute_query_and_commit(config, alter_query_out)

#############################
# Create PyG Data Object 
#############################
def create_pyg_data_object(config: configparser.ConfigParser, test_bool: bool=False) -> Data:
        """_summary_
        get the data from the database and create the PyG Data Object
        x: node features, y: node labels, edge_index: edge list, train_mask: boolean tensor with all ones, test_mask: boolean tensor with all ones
        Args:
            config (configparser.ConfigParser): _description_
            test_bool (bool, optional): _description_. Defaults to False.

        Returns:
            Data: _description_
        """
        x = create_x(config, test_bool)
        y = create_y(config, test_bool)
        edge_index = create_edge_index(config, test_bool)

        #todo add train and test split/mask 
        # Create a new boolean tensor with the same number of rows as 'existing_tensor' filled with True values
        train_mask = torch.ones(x.size(0), dtype=bool)
        test_mask = train_mask

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
        return data


def create_x(config: configparser.ConfigParser, test_bool: bool=False) -> torch.Tensor:
        """_summary_
        create the node features tensor x for the PyG Data Object
        Args:
            config (configparser.ConfigParser): _description_
            test_bool (bool, optional): _description_. Defaults to False.

        Returns:
            torch.Tensor: _description_
        """
        #edges = config['SQL']['table_name_edges']
        edges = config['threatrace']['test_edges']
        if test_bool:
                table_nodes = config['threatrace']['test_nodes']
                table_edges = config['threatrace']['test_edges']
        else:
                table_edges = config['threatrace']['train_edges']
                table_nodes = config['threatrace']['train_nodes']
        query_x = f"SELECT id FROM {table_nodes};"
        query_unique_event_types = f"SELECT DISTINCT event_type FROM {edges} ORDER BY event_type ASC;" # here "same source, so that every one have the same feature size"
        query_node_feature_out = f"SELECT src_num_new, event_type, COUNT(*) FROM {table_edges} GROUP BY src_num_new, event_type;"
        query_node_feature_in = f"SELECT dst_num_new, event_type, COUNT(*) FROM {table_edges} GROUP BY dst_num_new, event_type;"
        #whole runtime 3m 5s
        # Construct x -> finish
        results_query_x = execute_query(config, query_x)
        values = [row[0] for row in results_query_x]
        # Convert the values into a numpy array
        results_query_unique_event_types = execute_query(config, query_unique_event_types)
        # logging.info(f"results_query_unique_event_types: {results_query_unique_event_types}")
        # logging.info(f"Lenght of results_query_unique_event_types: {len(results_query_unique_event_types)}")
        event_types = [x[0] for x in results_query_unique_event_types]
        event_types_whole = [x + "_out" for x in event_types] + [x + "_in" for x in event_types]
        # logging.info(len(event_types_whole))
        df = pd.DataFrame(values, columns= ['id'], )
        for event_type in event_types_whole:
                 df[event_type] = 0
        ##
        results_query_node_feature_out = execute_query(config, query_node_feature_out)
        values = [row for row in results_query_node_feature_out]
        info_dict = {}
        for item in values:
                id_val, event_type, value = item
                if id_val not in info_dict:
                        info_dict[id_val] = {}
                info_dict[id_val][event_type] = value
        for event_type in set([item[1] for item in values]):
                event_type_orig = event_type
                event_type = event_type + "_out"
                df[event_type] = df['id'].map(lambda x: info_dict.get(x, {}).get(event_type_orig, 0))
        #runtime 1m 17s
        results_query_node_feature_in = execute_query(config, query_node_feature_in)
        values = [row for row in results_query_node_feature_in]
        info_dict = {}
        for item in values:
                id_val, event_type, value = item
                if id_val not in info_dict:
                        info_dict[id_val] = {}
                info_dict[id_val][event_type] = value
        for event_type in set([item[1] for item in values]):
                event_type_orig = event_type
                event_type = event_type + "_in"
                df[event_type] = df['id'].map(lambda x: info_dict.get(x, {}).get(event_type_orig, 0))
        df = df.drop(columns=['id'])
        # logging.info(f"df.shape: {df.shape}")
        # logging.info(f"df.columns: {df.columns}")
       #print(df[1].unique())
        # Convert DataFrame values to a NumPy array
        # Convert DataFrame values to a NumPy array
        numpy_array = df.values
        # Convert NumPy array to a PyTorch tensor
        x_tensor = torch.tensor(numpy_array, dtype=torch.float)
        return x_tensor

def create_y(config: configparser.ConfigParser, test_bool: bool=False) -> torch.Tensor:
        """_summary_
        create the node labels tensor y for the PyG Data Object
        Args:
            config (configparser.ConfigParser): _description_
            test_bool (bool, optional): _description_. Defaults to False.

        Returns:
            torch.Tensor: _description_
        """
        if test_bool:
                table_nodes = config['threatrace']['test_nodes']
        else:
                table_nodes = config['threatrace']['train_nodes']
        query_y = f"SELECT node_type_mapped FROM {table_nodes};"
        # Construct y  -> finish 
        results_query_y = execute_query(config, query_y)
        values = [row[0] for row in results_query_y]

        # Convert the values into a numpy array
        numpy_array = np.array(values)

        # Get the unique values and their counts
        unique_values, counts = np.unique(numpy_array, return_counts=True)
        # Print the unique values and their counts
        for value, count in zip(unique_values, counts):
                logging.info(f"Value: {value}, Count:, {count}")
        # Print the length of the array
        #logging.info("Length:", len(numpy_array))



        # Convert the numpy array into a tensor (using PyTorch)
        y_tensor = torch.tensor(numpy_array, dtype=torch.long)
        return y_tensor

def create_edge_index(config: configparser.ConfigParser, test_bool:bool=False) -> torch.Tensor:
        """_summary_
                create the edge list tensor edge_index for the PyG Data Object
        Args:
            config (configparser.ConfigParser): _description_
            test_bool (bool, optional): _description_. Defaults to False.

        Returns:
            torch.Tensor: _description_
        """
        if test_bool:
                table_edges = config['threatrace']['test_edges']
        else:
                table_edges = config['threatrace']['train_edges']
        query_edge_index_src = f"SELECT src_num_new FROM {table_edges};"
        query_edge_index_dst = f"SELECT dst_num_new FROM {table_edges};"

        results_query_edge_index_src = execute_query(config, query_edge_index_src)

        values = [row[0] for row in results_query_edge_index_src]

        # Convert the values into a numpy array
        numpy_array = np.array(values)

        # Print the length of the array
        #logging.info("Length:", len(numpy_array))
        # Get the unique values and their counts
        #unique_values, counts = np.unique(numpy_array, return_counts=True)
        # Print the unique values and their counts
        # for value, count in zip(unique_values, counts):
        #         print("Value:", value, "Count:", count)

        # Convert the numpy array into a tensor (using PyTorch)
        #edge_src_tensor = torch.from_numpy(numpy_array)
        edge_src = numpy_array
        # Decrease the value of every entry by one
        edge_src = edge_src - 1
        results_query_edge_index_dst = execute_query(config, query_edge_index_dst)

        values = [row[0] for row in results_query_edge_index_dst]

        # Convert the values into a numpy array
        numpy_array = np.array(values)

        # Print the length of the array
        logging.info(f"Length:, {len(numpy_array)}")
        # Get the unique values and their counts
        #unique_values, counts = np.unique(numpy_array, return_counts=True)
        # Print the unique values and their counts
        # for value, count in zip(unique_values, counts):
        #         print("Value:", value, "Count:", count)

        # Convert the numpy array into a tensor (using PyTorch)
        #edge_dst_tensor = torch.from_numpy(numpy_array)
        edge_dst = numpy_array
        # Decrease the value of every entry by one
        edge_src = edge_src - 1
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        return edge_index

def find_mapping_value_for_gt_uuid(config: configparser.ConfigParser, uuids: set) -> set:
        """_summary_
                Utility function to find the mapping value for the ground truth uuids
                The mapping value is the id of the node in the database
                This function is needed because the uuids in the gt are in a different format 
        Args:
            config (configparser.ConfigParser): _description_
            uuids (set): _description_

        Returns:
            set: _description_
        """
        table_nodes = config['threatrace']['test_nodes']
        gt_mapping = set()
        uuid_list = "','".join(uuids)
        query = f"SELECT id, node FROM {table_nodes} WHERE node IN ('{uuid_list}');"
        results_query = execute_query(config, query)
        gt_mapping = set(row[0] for row in results_query)
        gt_mapping_translate = set(row[1] for row in results_query)
        return gt_mapping, gt_mapping_translate

class MyPyGDataset(InMemoryDataset):
        """_summary_
        creates the PyG Dataset 
        it is used to load the data from the database and create the PyG Data Object
        if the data is created once it is saved in the output directory and loaded from there
        Args:
            InMemoryDataset (_type_): _description_
        """
        def __init__(self, config: configparser.ConfigParser, own_timestamp_percent:int, extension: str='',  transform: Optional[Callable] = None,
                     pre_transform: Optional[Callable] = None, **kwargs,):
                self.config = config
                self.name = "test"
                self.datasetname = config['threatrace']['test_edges'] + extension 
                self.datasetname = config['threatrace']['train_edges'] + extension
                self.output_dir = self.config["Directories"]["output"]
                self.own_timestamp_percent = own_timestamp_percent
                #----#
                super().__init__(self.output_dir, transform, pre_transform)
                self.data, self.slices = torch.load(self.processed_paths[0])

        
        @property
        def raw_file_names(self) -> str | List[str] | Tuple:
                return self.datasetname + '.pt'

        @property
        def processed_file_names(self) -> str | List[str] | Tuple:
                return self.datasetname + '.pt'
        
        def download(self):
                pass

        def transform(self):
                pass

        def process(self):
                create_threatrace_edge_list_for_both(self.config, self.own_timestamp_percent) # 4min
                create_threatrace_edge_list(self.config, False)
                create_threatrace_edge_list(self.config, True)
                data_test = create_pyg_data_object(self.config, True) #runtime 4m # Length = 12082603
                data_train = create_pyg_data_object(self.config, False) #runtime 4m Length = 12192401
                data = [data_train, data_test]
                torch.save(self.collate([data]), self.processed_paths[0])

# Create a type for MyClass
MyPyGDatasetType = Type[MyPyGDataset]
