# This file contains functions to preprocess DARPA data.
# The DARPA data is provided as a tar.gz file.
# The file contains a folder with json files.
# The json files are converted to csv files.
# The csv files are updated with a node mapping.
# The node mapping is created by iterating over the json files.
# The node mapping is used to replace the UUIDs in the csv files with a numeration.
# The numeration is used to create a graph with the mariadb import.
# The mariadb import requires a csv file with a header.
# The header contains the column names of the csv file.
# The column names are extracted from the json files.

import tarfile
import logging 
import glob
import json
import os
from typing import List, Set, Tuple, Any
import csv
from benchmark.constants import *
import shutil
import os
import re
import configparser


######
# Extracts a tar.gz file to a specified path.
def unzip_tar_gz(file_path: str, extract_path: str):
    """_summary_
        Extracts a tar.gz file to a specified path.
    Args:
        file_path (str): _description_
        extract_path (str): _description_
    """
    logging.info(f"Extracting '{file_path}' to '{extract_path}'.")
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(extract_path)
        logging.info(f"Successfully extracted '{file_path}' to '{extract_path}'.")
    except tarfile.TarError as e:
        logging.error(f"Error extracting '{file_path}': {e}")

######
def rename_files(folder_path: str):
    """
    Renames files in a folder.
    The files are renamed by adding a leading zero to the file name.
    The leading zero is added if the file name ends with a single digit.
    Args:
        folder_path (str): _description_
    """
    # Specify the directory path
    logging.info(f"Renaming files in '{folder_path}'.")
    directory = folder_path

    # List all files in the directory
    files = os.listdir(directory)
    # Define the regular expression pattern
    pattern = r'(\.\d+)$'
    # Process each file
    for file in files:
        # Check if the file name ends with a single digit
        match = re.search(pattern, file)
        if match:
            # Get the matched part (e.g., '.1', '.10')
            matched_part = match.group(1)
            
            # Check if the matched part has only one digit
            if len(matched_part) == 2:
                # Rename the file by adding a leading zero
                new_filename = re.sub(pattern, '.0' + matched_part[1], file)
                
                # Get the current and new file paths
                current_path = os.path.join(directory, file)
                new_path = os.path.join(directory, new_filename)
                
                # Rename the file
                os.rename(current_path, new_path)
                
                logging.info(f"Renamed '{file}' to '{new_filename}'")

######
# Finds all json files in a specified folder.
def find_json_files(folder_path: str) -> list:
    """_summary_
        Finds all json files in a specified folder.
        Returns a list of file paths.
    Args:
        folder_path (str): _description_

    Returns:
        list: _description_
    """
    logging.info(f"Searching for json files in '{folder_path}'.")
    file_pattern = '*.json*'

    json_files = glob.glob(f"{folder_path}/{file_pattern}")

    json_files.sort()
    return json_files

######
# get header columns for csv file
def get_keys(json_data: Any, parent_key: str='', key_set=set()) -> set:
    """summary
        Extracts the keys from a json file.
        Returns a set of keys.
    Args:
        json_data (Any): _description_  
        parent_key (str, optional): _description_. Defaults to ''.
        key_set (set, optional): _description_. Defaults to set().
    Returns:
        set: _description_
    """
    for key, value in json_data.items():
        flattened_key = f"{parent_key}.{key}" if parent_key else key
        key_set.add(flattened_key)
        if isinstance(value, dict):
            get_keys(value, flattened_key, key_set)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    get_keys(item, flattened_key, key_set)
    return key_set

def iterate_over_files(file_list: str) -> Tuple[Set, Set]:
    """summary
        Iterates over a list of files.
        Extracts the keys from the json files.
        Returns a set of keys.
    Args:
        file_list (str): _description_
    Returns:
        Tuple[Set, Set]: _description_
    """
    logging.info("Iterating over files")
    keys = set()
    line_count = 0
    for filename in file_list:
        logging.info(f"Processing file {filename}")
        filepath = os.path.join(filename)
        with open(filepath, 'r') as f:
            for line in f:
                line_count += 1
                # Parse the line as JSON
                if line_count % 1000000 == 0:
                    logging.info(f"Processed {line_count} lines")
                data = json.loads(line)

                # Extract the keys from the JSON data
                keys.update(get_keys( data))
    return keys

def write_keys(keys, output_file) -> None:
    """summary
        Writes the keys to a file.
    Args:
        keys (set): _description_
        output_file (str): _description_
    """
    with open(output_file, 'w') as f:
        for key in sorted(keys):
            f.write(key + '\n')

def split_keys(keys) -> Tuple[Set, Set]:
    """summary
        Splits the keys into node and edge keys.
        Returns a tuple of node and edge keys.
    Args:
        keys (set): _description_
    Returns:
        Tuple[Set, Set]: _description_
    """
    node_keys = set()
    event_keys = set()
    for key in keys:
        if key.startswith('datum.com.bbn.tc.schema.avro.cdm18.Event'):
            event_keys.add(key)
        else:
            node_keys.add(key)
    return node_keys, event_keys

def get_and_write_keys(file_list:list[str]) -> Tuple[Set, Set]:
    """summary
        Gets the keys from the json files.
        Splits the keys into node and edge keys.
        Writes the node and edge keys to files.
        Returns a tuple of node and edge keys.
    Args:
        file_list (list[str]): _description_

    Returns:
        Tuple[Set, Set]: _description_
    """
    keys = iterate_over_files(file_list)
    node_keys, event_keys = split_keys(keys)
    return node_keys, event_keys
######
# write node and edge csv files from json files with header
def write_row(data, writer, keys):
    row = [data.get(key, '') for key in keys]
    writer.writerow(row)
    
def flatten_json(nested_json, sep='.') -> dict:
    flattened_dict = {}
    def flatten(inner, outer_key=''):
        if isinstance(inner, dict):
            for key, value in inner.items():
                flatten(value, outer_key + sep + key if outer_key else key)
        elif isinstance(inner, list):
            for i, value in enumerate(inner):
                flatten(value, outer_key + sep + str(i) if outer_key else str(i))
        else:
            # Check if value has a comma and replace it with a colon # important for csv files -> especially for mariadb import
            if isinstance(inner, str) and ',' in inner:
                inner = inner.replace(',', ':')
            if isinstance(inner, str) and '\n' in inner:
                inner = inner.replace('\n', 'BS')
                #print(f"Commata found: {inner}")
            flattened_dict[outer_key] = inner
    flatten(nested_json)
    return flattened_dict

def process_json_files(file_list: List[str], output_dir: str, node_keys: Set[str], edge_keys: Set[str]) -> None:
    """summary
        Processes a list of json files.
        Extracts the keys from the json files.
        Writes the keys to a file.
    Args:
            file_list (List[str]): _description_
            output_dir (str): _description_
            node_keys (Set[str]): _description_
            edge_keys (Set[str]): _description_
    """
    logging.info("Processing json files")
    line_count = 0
    event_file = output_dir + 'edges.csv'
    node_file = output_dir + 'nodes.csv'

    with open(event_file, 'w', newline='') as event_file, \
         open(node_file, 'w', newline='') as node_file:
    #open(output_file, 'w', newline='') as file, \
         
        #writer = csv.writer(file)
        event_writer = csv.writer(event_file)
        non_event_writer = csv.writer(node_file)

        #writer.writerow(keys)
        event_writer.writerow(edge_keys)
        non_event_writer.writerow(node_keys)

        for filepath in file_list:
            with open(filepath, 'r') as f:
                logging.info(f"Processing file {filepath}")
                for line in f:
                    line_count += 1
                    if line_count % 100000 == 0:
                        logging.info(f"Processed {line_count} lines")
                    try:
                        data = json.loads(line.strip())
                        try:
                            data_processed = flatten_json(data)
                            if EVENT in data_processed:
                                write_row(data_processed, event_writer, edge_keys)
                            else:
                                write_row(data_processed, non_event_writer, node_keys)
                            #write_row(data_processed, writer, keys)
                        except Exception as e: 
                            logging.error(e)
                            continue
                    except Exception as e: 
                        logging.error(e)
                        continue
    event_file.close()
    node_file.close()
###### 
#add node mapping to nodes and edges 
# Function to search for the column index of UUID in a CSV file
def find_uuid_column_index(csv_file, node=False):
    """_summary_
        Searches for the column index of UUID in a CSV file.
        Returns a dictionary of column names and column indices.
    Args:
        csv_file (_type_): _description_
        node (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    logging.info(f'Finding uuid column index in {csv_file}')
    uuid_column_indes_list = dict()
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        for index, column_name in enumerate(header):
            if node:
                for node_column in NODE_COLUMNS:
                    if node_column in column_name:
                        logging.info(f"Found node column index: {index}")
                        uuid_column_indes_list[column_name] = index
            else:
                 for edge_column in EVENT_COLUMNS:
                    if edge_column in column_name:
                        logging.info(f"Found node column index: {index}")
                        uuid_column_indes_list[column_name] = index
    return uuid_column_indes_list

def get_node_mapping(output_dir:str, nodes_file:str) -> Tuple[dict, dict]:
    """_summary_
        Gets the node mapping from a CSV file.
        Returns a tuple of the uuid column index and the node mapping.
        nodes_uuid_column_index is a dict with entries like: {'datum.com.bbn.tc.schema.avro.cdm18.Host.uuid': 0}
        node_mapping is adict with entries like:{'00000000-0000-0000-0000-000000000000': 0}
    Args:
        output_dir (str): _description_
        nodes_file (str): _description_

    Returns:
        Tuple[dict, dict]: _description_
    """
    logging.info(f'Getting node mapping from {nodes_file}')
    # Define the CSV file paths
    nodes_uuid_column_index = find_uuid_column_index(nodes_file, node=True)

    # Read and map the nodes
    node_mapping = {}
    node_numeration = 0 # Start with 0

    with open(nodes_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row

        for row in reader:
            #check which entry is not nan and use this as uuid
            for column_name, column_index in nodes_uuid_column_index.items():
                if row[column_index] != '':
                    node_uuid = row[column_index]
                    break
            if node_uuid not in node_mapping:
                node_mapping[node_uuid] = node_numeration
                node_numeration += 1
    # runtime ~10sec
    logging.info(f'Found {len(node_mapping)} nodes')
    node_mapping_file = output_dir + 'node_mapping.csv'
    with open(node_mapping_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['UUID', 'Numeration'])
        for node_uuid, node_numeration in node_mapping.items():
            writer.writerow([node_uuid, node_numeration])
    return nodes_uuid_column_index, node_mapping

# also further UUDS mappings could be placed here, e.g. Principals Src/Sink Objects
def update_node_file(output_dir:str, nodes_file:str, nodes_uuid_column_index: dict, node_mapping:dict):
    """_summary_
        Updates a node file with a node mapping.
        Writes the updated node file to a new file.
        old node mapping is a uuid and new is a numeration
    Args:
        output_dir (str): _description_
        nodes_file (str): _description_
        nodes_uuid_column_index (dict): _description_
        node_mapping (dict): _description_
    """
    logging.info(f'Updating node file {nodes_file}')
    #runtime 8sec
    updated_node_file = output_dir + 'updated_nodes.csv'
    with open(updated_node_file, 'w', newline='') as file:
        writer = csv.writer(file)
        with open(nodes_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Read the header row
            #updated_edges.append(header + ['Source Numeration', 'Destination Numeration', 'Destination2 Numeration'])
            writer.writerow(header + ["node_type", "collected_uuid", 'UUID_Numeration', 'End_Column'])
            counter = 0
            for row in reader:
                counter += 1
                if counter % 1000000 == 0:
                    logging.info(f"Processed {counter} edges")
                for column_name, column_index in nodes_uuid_column_index.items():
                    if row[column_index] != '':
                        column_name = column_name.replace('datum.com.bbn.tc.schema.avro.cdm18.', '')
                        column_name = column_name.replace('.uuid', '')
                        node_uuid = row[column_index]
                        node_type = column_name
                        break

                uuid_numeration = node_mapping.get(node_uuid)
                collected_uuid = node_uuid
                end_column = 0 # important for mariadb import
                writer.writerow(row + [node_type, collected_uuid, uuid_numeration, end_column])
    logging.info(f'Updated node file {nodes_file}')
    logging.info(f'Wrote updated node file to {updated_node_file}')

def update_edge_file(output_dir: str, edges_file: str, edges_uuid_column_index:dict, node_mapping:dict):
    """_summary_
        Updates an edge file with a node mapping.
        Writes the updated edge file to a new file.
        old node mapping is a uuid and new is a numeration
    Args:
        output_dir (str): _description_
        edges_file (str): _description_
        edges_uuid_column_index (dict): _description_
        node_mapping (dict): _description_
    """
    # Read the edges and map the UUIDs to the generated numeration
    #runtime 2min 
    #updated_edges = []
    updated_edges_file = output_dir + 'updated_edges.csv'
    with open(updated_edges_file, 'w', newline='') as file:
        writer = csv.writer(file)
        with open(edges_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Read the header row
            #updated_edges.append(header + ['Source Numeration', 'Destination Numeration', 'Destination2 Numeration'])
            writer.writerow(header + ['Source_Numeration', 'Destination_Numeration', 'Destination2_Numeration', 'End_Column'])
            counter = 0
            for row in reader:
                counter += 1
                if counter % 1000000 == 0:
                    logging.info(f"Processed {counter} edges")
                for column_name, column_index in edges_uuid_column_index.items():
                    if column_name == EVENT_SUBJECT:
                        source_uuid = row[column_index]
                    if column_name == EVENT_OBJECT:
                        destination_uuid = row[column_index]
                    if column_name == EVENT_OBJECT2:
                        destination2_uuid = row[column_index]
                

                # source_uuid = row[10]
                # destination_uuid = row[33]
                # destination2_uuid = row[20]

                source_numeration = node_mapping.get(source_uuid)
                destination_numeration = node_mapping.get(destination_uuid)
                destination2_numeration = node_mapping.get(destination2_uuid)
                end_column = 0 # important for mariadb import

                #updated_edges.append(row + [source_numeration, destination_numeration, destination2_numeration])
                writer.writerow(row + [source_numeration, destination_numeration, destination2_numeration, end_column])
    logging.info(f'Updated edge file {edges_file}')
    logging.info(f'Wrote updated edge file to {updated_edges_file}')
######

# whole preprocessing process
# runtime (eng3 cadets first attack) ~10min
def process_darpa_data(config: configparser.ConfigParser):
    """_summary_
        Manages the preprocessing of DARPA data.
        1. Unzip tars.
        2. Rename files. (for ordering of files)
        3. Find all json files in folder.
        4. Get keys from json files.
        5. Write csv files from json files.
        6. Get node mapping and uuid column index.
        7. Get uuid column index for edges.
        8. Update node file with uuid - id mapping.
        9. Update edge file with uuid - id mapping.
        10. Delete unzipped folder.
        11. Delete old node and edge files.
    Args:
        config (configparser.ConfigParser): _description_
    """
    input_file = config['Files']['tar']
    output_dir = config['Directories']['output']
    unzipped_dir = config['Directories']['unzipped']
    #create folder, if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f'Created output folder {output_dir}')
    # unzip tar.gz file
    unzip_tar_gz(input_file, unzipped_dir)
    logging.info(f'Unzipped tar.gz file to {unzipped_dir}')
    rename_files(unzipped_dir)
    logging.info(f'Renamed files in {unzipped_dir}')
    # find all json files in folder
    file_list = find_json_files(unzipped_dir)
    logging.info(f'Found {len(file_list)} json files in {unzipped_dir}')
    # get keys from json files
    node_keys, edge_keys = get_and_write_keys(file_list)
    logging.info(f'Found {len(node_keys)} node keys and {len(edge_keys)} edge keys')
    # write csv files from json files
    process_json_files(file_list, output_dir, node_keys, edge_keys)
    logging.info(f'Wrote csv files to {output_dir}')
    nodes_file = output_dir + 'nodes.csv'
    edges_file = output_dir + 'edges.csv'
    # get node mapping and uuid column index
    nodes_uuid_column_index, node_mapping = get_node_mapping(output_dir, nodes_file)
    # get uuid column index for edges
    edges_uuid_column_index =  find_uuid_column_index(edges_file)
    # update node and edge file and sace to new file updated_nodes.csv and updated_edges.csv
    update_node_file(output_dir, nodes_file, nodes_uuid_column_index, node_mapping)
    update_edge_file(output_dir, edges_file, edges_uuid_column_index, node_mapping)
    logging.info(f'Updated node and edge files with node mapping')
    # delete unzipped folder
    shutil.rmtree(unzipped_dir)
    logging.info(f'Deleted unzipped folder {unzipped_dir}')
    # delete old node and edge files
    os.remove(nodes_file)
    os.remove(edges_file)
    logging.info(f'Deleted old node and edge files')
######