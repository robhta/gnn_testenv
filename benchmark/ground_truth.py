
from typing import Dict, List, Tuple, Any
import logging
from benchmark.utils import *
from benchmark.ground_truth import *
from benchmark.construct_threatrace_graph import *
import yaml
import os
import pandas as pd
import configparser
#from benchmark.db_import import *


def read_gt_yaml(config: configparser.ConfigParser) -> Dict:
    """_summary_
        Reads the ground truth yaml file and returns the data as a dictionary.
    Args:
        config (configparser.ConfigParser): _description_

    Returns:
        Dict: _description_
    """

    logging.info(f"read_gt_yaml:")
    file_path = config['Files']['ground_truth']
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    logging.debug(yaml_data)
    return yaml_data

def read_gt_yaml_with_filepath(file_path: str) -> Dict:
    """_summary_
        Reads the ground truth yaml file and returns the data as a dictionary.
    Args:
        file_path (str): _description_

    Returns:
        Dict: _description_
    """
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    logging.debug(yaml_data)
    return yaml_data

def extract_time_information(yaml_data: Dict) -> Tuple[str, str, str]:
    """_summary_
        Extracts the time information from the yaml file.
    Args:
        yaml_data (Dict): _description_

    Returns:
        Tuple[str, str, str]: _description_
    """
    date = yaml_data['date']
    start_ts = yaml_data['start_time']
    end_ts = yaml_data['end_time']
    return date, start_ts, end_ts

def extract_search_terms(yaml_data: Dict) -> List[str]:
    """_summary_
        Extracts the search terms from the yaml file.
    Args:
        yaml_data (Dict): _description_
    
    Returns:
        List[str]: _description_
    """
    search_terms = yaml_data['search_terms']
    return search_terms

def scan_edges(config: configparser.ConfigParser, yaml_data: Dict) -> List[Tuple]:
    """_summary_
        Scans the edges table for the given time range and search terms.
        The result of the scan is a list of tuples, where each tuple contains the uuids of the subject, predicate and object of the edge.
    Args:
        config (configparser.ConfigParser): _description_
        yaml_data (Dict): _description_
    Returns:
        List[Tuple]: _description_
    """
    date, start_ts, end_ts = extract_time_information(yaml_data)
    # Define the time range
    start_time_unix_ns = parse_darpa_gt_to_unix_ns(date=date, time=start_ts, plus=False)
    end_time_unix_ns = parse_darpa_gt_to_unix_ns(date=date, time=end_ts, plus=True)
    # Define the search strings
    search_terms = extract_search_terms(yaml_data)
    logging.debug(search_terms)
    #14 secs 
    connection = establish_connection(config)
    # Create a cursor object to interact with the database
    cursor = connection.cursor()

    # Get the list of column names from the table
    table_name = config['SQL']['table_name_edges']
    cursor.execute(f"SHOW COLUMNS FROM {table_name}")
    columns = [column[0] for column in cursor.fetchall()]
    logging.debug(columns)
    # Construct the WHERE clause for time range
    time_column = 'event_timestampnanos'  # Replace with the actual time column name
    time_range_condition = f"{time_column} BETWEEN '{start_time_unix_ns}' AND '{end_time_unix_ns}'"

    # Construct the WHERE clause to search each column for each search string
    where_clauses = []
    for search_term in search_terms:
        string_clauses = [f"{column} LIKE '%{search_term}%'" for column in columns]
        where_clauses.append('(' + ' OR '.join(string_clauses) + ')')

    # Combine the time range condition and search conditions
    where_clause = f"{time_range_condition} AND ({' OR '.join(where_clauses)})"

    # Construct the SELECT query
    query = f"""SELECT 
                    event_subject_com_bbn_tc_schema_avro_cdm18_uuid, 
                    event_predicateobject_com_bbn_tc_schema_avro_cdm18_uuid,
                    event_predicateobject2_com_bbn_tc_schema_avro_cdm18_uuid 
                FROM {table_name} 
                WHERE {where_clause}"""

    # Execute the query
    cursor.execute(query)
    # Fetch the results
    results = cursor.fetchall()
    

    # Close the cursor and connection
    cursor.close()
    connection.close()
    return results

def scan_nodes(config: configparser.ConfigParser, yaml_data: Dict) -> List[Tuple]:
    """_summary_
        Scans the nodes table for the given time range and search terms.
        The result of the scan is a list of tuples, where each entry represents a node uuid.
    Args:
        config (configparser.ConfigParser): _description_
        yaml_data (Dict): _description_

    Returns:
        List[Tuple]: _description_
    """
    logging.info(f"scan_nodes:")
        #14secs 
    connection = establish_connection(config)
    # Create a cursor object to interact with the database
    cursor = connection.cursor()

    # Define the search strings
    search_terms = extract_search_terms(yaml_data)
    logging.info(f"search_terms: {search_terms}")

    # Get the list of column names from the table
    table_name = config['SQL']['table_name_nodes']
    cursor.execute(f"SHOW COLUMNS FROM {table_name}")
    columns = [column[0] for column in cursor.fetchall()]
    logging.info(f"columns: {columns}")
    # Construct the WHERE clause for time range
    #time_column = 'event_timestampnanos'  # Replace with the actual time column name
    #time_range_condition = f"{time_column} BETWEEN '{start_time}' AND '{end_time}'"

    # Construct the WHERE clause to search each column for each search string
    where_clauses = []
    for search_term in search_terms:
        string_clauses = [f"{column} LIKE '%{search_term}%'" for column in columns]
        where_clauses.append('(' + ' OR '.join(string_clauses) + ')')

    # Combine the time range condition and search conditions
    where_clause = f"({' OR '.join(where_clauses)})"

    # Construct the SELECT query
    query = f"SELECT collected_uuid FROM {table_name} WHERE {where_clause}"
    logging.debug(f"query: {query}")
    # Execute the query
    cursor.execute(query)
    # Fetch the results
    results = cursor.fetchall()
    logging.info(f"results length: {len(results)}")

    # Close the cursor and connection
    cursor.close()
    connection.close()
    #with a time short time range it still takes: 30sec -> without index on event_timestampnanos
    return results

def get_atk_nodes_from_yaml(config: configparser.ConfigParser, 
                            yaml_data: Dict, 
                            with_edges: bool = True, 
                            only_subject: bool = False) -> set:
    """_summary_
        Wrapper over scan_edges and scan_nodes to get the set of nodes that are involved in the ground truth.
        The result is a set of node uuids.
        The set represents all nodes that are involved in the ground truth, based on the time and search value conditions.
    Args:
        config (configparser.ConfigParser): _description_
        yaml_data (Dict): _description_
        with_edges (bool, optional): _description_. Defaults to True.
        only_subject (bool, optional): _description_. Defaults to False.
    Returns:   
        set: _description_
    """
    logging.info(f"get_atk_nodes_from_yaml:")
    node_set = set()
    logging.info("Scan edges")
    results_in_edges = scan_edges(config, yaml_data)
    #with a time short time range it still takes: 30sec -> without index on event_timestampnanos
    # with index on event_timestampnanos it takes 
    logging.info(len(results_in_edges))
    node_set = set()
    if with_edges:
        if only_subject:
            for row[0] in results_in_edges:
                last_three_columns = row  # Extract the important last three columns using slicing # outdated 
                node_set.update(last_three_columns)
        else:
            for row in results_in_edges:
                last_three_columns = row  # Extract the important last three columns using slicing # outdated 
                node_set.update(last_three_columns)
        logging.info(f"Length of node_set: {len(node_set)}")
    logging.info("Scan nodes")
    results_in_nodes = scan_nodes(config, yaml_data)
        #14secs 
    len(results_in_nodes)
    for row in results_in_nodes:
        last_three_columns = row  # Extract the important last three columns using slicing 
        node_set.update(last_three_columns)
    logging.info(f"Length of node_set: {len(node_set)}") #+ 2 from nodes
    return node_set

def get_atk_nodes(config: configparser.ConfigParser) -> set:
    """_summary_
        Wrapper over get_atk_nodes_from_yaml to get the set of nodes that are involved in the ground truth.
    Args:
        config (configparser.ConfigParser): _description_

    Returns:
        set: _description_
    """
    logging.info(f"get_atk_nodes")
    node_set = set()
    file_path = config['Directories']['output'] + "atk_nodes.txt"
    if os.path.exists(file_path):
        logging.info("File exists.")
        with open(file_path, 'r') as file:
            for line in file:
                node_set.add(line.strip())
    else:
        logging.info(f"Getting atk nodes from config: {config}")
        yaml_data = read_gt_yaml(config)
        logging.info(f"yaml_data: {yaml_data}")
        node_set = get_atk_nodes_from_yaml(config, yaml_data, with_edges = True, only_subject=False) ## here TRUE OR FALSE FOR GT 
        #logging.info(f"node_set: {node_set}")
        with open(file_path, 'w') as file:
            file.writelines('\n'.join(node_set))
    return node_set

def get_atk_nodes_with_multiple_gts(config: configparser.ConfigParser, ground_truths: List[str]) -> set:
    """_summary_
        Wrapper over get_atk_nodes_from_yaml for multiple gts to get the set of nodes that are involved in the ground truth.
    Args:
        config (configparser.ConfigParser): _description_
        ground_truths (List[str]): _description_

    Returns:
        set: _description_
    """
    logging.info(f"get_atk_nodes_with_multiple_gts")
    whole_node_set = set()
    file_path = config['Directories']['output'] + "atk_nodes.txt"
    if os.path.exists(file_path):
        logging.info("File exists.")
        with open(file_path, 'r') as file:
            for line in file:
                whole_node_set.add(line.strip())
    else:
        for gt in ground_truths:
            yaml_data = read_gt_yaml_with_filepath(gt)
            node_set = get_atk_nodes_from_yaml(config, yaml_data)
            logging.info(len(node_set))
            whole_node_set.update(node_set)
    return whole_node_set

def get_gt_information(config: configparser.ConfigParser, gt_with_numbers: List[int]) -> pd.DataFrame:
    """_summary_
        Gets the information of the ground truth nodes from the nodes table.
    Args:
        config (configparser.ConfigParser): _description_
        gt_with_numbers (List[int]): _description_
    
    Returns:
        pd.DataFrame: _description_"""
    logging.info(f"get_gt_information")
    nodes_table = config['SQL']['table_name_nodes']
    query_columns = f"SHOW COLUMNS FROM {nodes_table}"
    results_query_columns = execute_query(config, query_columns)
    gt_with_numbers_incremented = [x + 1 for x in gt_with_numbers]
    query = f"SELECT * FROM {nodes_table} WHERE id IN {tuple(gt_with_numbers_incremented)}"
    results_query = execute_query(config, query)
    df = pd.DataFrame(results_query, columns=[x[0] for x in results_query_columns])
    return df

def get_atk_nodes_for_evaluation(config: configparser.ConfigParser, multi: bool=False,  gts: List[str]=None) -> List[int]:
    from benchmark.construct_threatrace_graph import find_mapping_value_for_gt_uuid
    """_summary_
        Gets the atk nodes for evaluation.
        1. Gets the atk nodes from the config file.
        2. Maps the atk nodes to the node ids in the nodes table.
        3. Decrements the node ids by 1 to get the node ids in the graph.
    Args:
        config (configparser.ConfigParser): _description_
        multi (bool, optional): _description_. Defaults to False.
        gts (List[str], optional): _description_. Defaults to None.
    Returns:
        List[int]: _description_
    """
    logging.info(f"get_atk_nodes_for_evaluation:")
    if multi:
        atk_nodes = get_atk_nodes_with_multiple_gts(config, gts)
    else:
        atk_nodes = get_atk_nodes(config)
    logging.info(f"atk_nodes: {len(atk_nodes)}")
    mapped_atk_nodes, translation = find_mapping_value_for_gt_uuid(config, atk_nodes)
    logging.info(f"mapped_atk_nodes: {len(mapped_atk_nodes)}")
    mapped_atk_nodes_decremented = [x - 1 for x in mapped_atk_nodes]
    return mapped_atk_nodes_decremented