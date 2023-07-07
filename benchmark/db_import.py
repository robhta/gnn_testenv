



# Description: This file contains functions to import data from csv files into a mariadb database


# Module Imports
import mysql.connector as mariadb
import json
import sys
import csv 
from typing import List, Dict, Tuple, Any, Optional
import logging
from benchmark.constants import * 
import configparser 
from benchmark.utils import *

def establish_connection(config: configparser.ConfigParser) -> mariadb.connection:
    """_summary_
        Establishes a connection to a MariaDB database.
        Values are read from the config file.
    Args:
        config (configparser.ConfigParser): _description_

    Returns:
        mariadb.connection: _description_
    """
    # Connect to MariaDB Platform
    logging.info("Establishing connection")
    user = config['SQL']['user']
    password = config['SQL']['password']
    host = config['SQL']['host']
    port = config['SQL']['port']
    database = config['SQL']['database']
    try:
        conn = mariadb.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database=database
        )
    except mariadb.Error as e:
        logging.error(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)
    logging.info("Connection established")
    return conn

def load_data_infile(conn: mariadb.connection, config: configparser.ConfigParser, edges: bool=False) -> None:
    """_summary_
        Loads data from a csv file into a MariaDB database.
        Values are read from the config file.
        The csv file has to be in the same directory as the script on the docker container.
        The Method is used, because it is much faster than inserting the data row by row.
    Args:
        conn (mariadb.connection): _description_ 
        config (configparser.ConfigParser): _description_
        edges (bool, optional): _description_. Defaults to False.
        """
    logging.info(f"load_data_infile:")
    cur = conn.cursor()
    # CSV Path on Docker
    #csv_file_docker = "/output/cadets/attack02/output/updated_edges.csv" # CARE IF IT IS SET FROM CONFIG -> IT HAS to be in " "
    csv_file_docker_nodes = config['Files']['csv_file_docker_nodes']
    csv_file_docker_edges = config['Files']['csv_file_docker_edges']
    # Define the table name where you want to import the CSV data
    table_name_nodes = config['SQL']['table_name_nodes']
    table_name_edges = config['SQL']['table_name_edges']

    if edges:
        table_name = table_name_edges
        csv_file_docker = str(csv_file_docker_edges)
    else:   
        table_name = table_name_nodes
        csv_file_docker = str(csv_file_docker_nodes)
    #csv_file_docker = "/output/cadets/attack01/output/updated_nodes.csv" # CARE IF IT IS SET FROM CONFIG -> IT HAS to be in " "
    # Define the SQL query to load the CSV data into the table
    #csv_file_docker = "/Users/robinbuchta/Documents/DARPA/cadets/attack01/output/updated_nodes.csv"
    # Get the column names and types from the table
    cur.execute(f"DESCRIBE {table_name}")
    columns_info = cur.fetchall()
    columns = [column[0] for column in columns_info if column[0] != 'id']
    column_types = {column[0]: column[1] for column in columns_info if column[0] != 'id'}

    # Generate the SET clause with NULLIF() for each INT column
    set_clauses = []
    load_columns = []
    for column in columns:
        if column_types[column].startswith('int') or column_types[column].startswith('bigint'):
            set_clauses.append(f"{column} = NULLIF(@{column}, '')") # set to NULL if empty string
            load_columns.append(f"@{column}")
        else:
            load_columns.append(column)
    set_clause_str = ", ".join(set_clauses)

    
    # load_query = f"""
    #     LOAD DATA INFILE '{csv_file_docker}'
    #     INTO TABLE {table_name}
    #     FIELDS TERMINATED BY ','  
    #     LINES TERMINATED BY '\n'   
    #     IGNORE 1 LINES
    #     ({', '.join(load_columns)})
    #     SET {set_clause_str}
    # """
    load_query = f"""
        LOAD DATA INFILE '{csv_file_docker}'
        INTO TABLE {table_name}
        COLUMNS TERMINATED BY ','     
        IGNORE 1 LINES
        ({', '.join(load_columns)})
        SET {set_clause_str}
    """

    # load_query = f"""
    #     LOAD DATA INFILE '{csv_file_docker}'
    #     INTO TABLE {table_name}
    #     FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY "'"
    #     LINES TERMINATED BY '\n'
    #     IGNORE 1 ROWS
    #     ({', '.join(load_columns)})
    # """
    # Execute the LOAD DATA INFILE query
    cur.execute(load_query)

def commit_and_close(conn: mariadb.connection) -> None:
    """_summary_
        Commits the changes and closes the connection to the database.
    Args:
        conn (mariadb.connection): _description_
    """
    logging.info("Committing and closing connection")
    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def create_table(conn: mariadb.connection, 
                config: configparser.ConfigParser, 
                columns_with_type: str, edges: bool=False) -> None:
    """_summary_
    Creates a table in the database with the given columns and data types.
    Args:
        conn (mariadb.connection): _description_
        config (configparser.ConfigParser): _description_
        columns_with_type (str): _description_
        edges (bool, optional): _description_. Defaults to False.
    """
    cur = conn.cursor()
    # Generate CREATE TABLE query
    table_name_nodes = config['SQL']['table_name_nodes']
    table_name_edges = config['SQL']['table_name_edges']
    logging.info(f"table_name_nodes: {table_name_nodes}")
    logging.info(f"table_name_edges: {table_name_edges}")
    if edges:
        table_name = table_name_edges
    else:   
        table_name = table_name_nodes
    # Drop the table if it exists
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    logging.info(f"droped table if exists: {table_name}")
    # Create the table
    create_table_query = f"""
        CREATE TABLE {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            {columns_with_type}                   # list of columns with data type (e.g. 'id INT, name TEXT'
        )
    """
    logging.debug(f"create_table_query: {create_table_query}")
    cur.execute(create_table_query)
    logging.info(f"created table: {table_name}")

    # Create an index for a column 
    # Todo: create a function for it 
    if edges:
        your_index_name = "ts_index"
        column = "event_timestampnanos"    
        create_index_query = f'''
        CREATE INDEX {your_index_name} ON {table_name} ({column})
        '''
        cur.execute(create_index_query)
        your_index_name = "subject_index"
        #column = "event_subject_com_bbn_tc_schema_avro_cdm18_uuid"   
        column = "source_numeration"
        create_index_query = f'''
        CREATE INDEX {your_index_name} ON {table_name} ({column})
        '''
        cur.execute(create_index_query)
        your_index_name = "predicateobject_index"
        #column = "event_predicateobject_com_bbn_tc_schema_avro_cdm18_uuid"   
        column = "destination_numeration"
        create_index_query = f'''
        CREATE INDEX {your_index_name} ON {table_name} ({column})
        '''
        cur.execute(create_index_query)
        your_index_name = "predicateobject2_index"
        #column = "event_predicateobject2_com_bbn_tc_schema_avro_cdm18_uuid"   
        column = "destination2_numeration"
        create_index_query = f'''
        CREATE INDEX {your_index_name} ON {table_name} ({column})
        '''
        cur.execute(create_index_query)
        logging.info('Index created successfully.')
    else:
        your_index_name = "uuid_index"
        #column = "collected_uuid"   
        column = "uuid_numeration"
        create_index_query = f'''
        CREATE INDEX {your_index_name} ON {table_name} ({column})
        '''
        cur.execute(create_index_query)


def prepare_columns(file:str, n: int=None) -> str:
    """_summary_
        Prepares the columns for the create_table function.
        1. Reads the file and extracts the column names and data types 
        2. Parses the dict data into a string
        3. Replaces some data types and column names (which resulted in errors)
        4. Returns the string with the columns and data types
    Args:
        file (str): _description_
        n (int, optional): _description_. Defaults to None.
    Returns:
        str: _description_"""
    logging.info(f"prepare_columns: {file}")
    columns_with_datatype = find_sql_table_type(file, n)
    logging.info(f"len of columns_with_datatype: {len(columns_with_datatype)}")
    columns_with_type, columns = parse_dict_to_string(columns_with_datatype)
    def replace_columns(columns: str):
        columns = columns.replace('.', '_')
        columns = columns.replace('datum_com_bbn_tc_schema_avro_cdm18_', '')
        columns = columns.replace('INT', 'INT NULL')
        columns = columns.lower()
        columns = columns.replace('int null', 'INT NULL')
        columns = columns.replace('subject_starttimestampnanos INT NULL', 'subject_starttimestampnanos BIGINT NULL')
        columns = columns.replace('event_timestampnanos INT NULL', 'event_timestampnanos BIGINT NULL')
        columns = columns.replace('event_properties_map_ret_msgid INT NULL', 'event_properties_map_ret_msgid BIGINT NULL')
        columns = columns.replace('memoryobject_memoryaddress INT NULL', 'memoryobject_memoryaddress BIGINT NULL')
        columns = columns.replace('subject_cmdline_string varchar(255)', 'subject_cmdline_string TEXT')
        columns = columns.replace('event_properties_map_cmdline varchar(255)', 'event_properties_map_cmdline TEXT')
        columns = columns.replace('event_properties_map_shmid INT NULL', 'event_properties_map_shmid BIGINT NULL')
        #columns = columns.replace('event_properties_map_port INT NULL', 'event_properties_map_port TEXT')
        #columns = columns.replace('subject_privilegelevel INT', 'subject_privilegelevel varchar(255)')
        return columns
    columns_with_type = replace_columns(columns_with_type)
    columns = replace_columns(columns)
    logging.info(f"columns_with_type: {columns_with_type}")
    logging.info(f"columns: {columns}")
    return columns_with_type 

def import_nodes(config: configparser.ConfigParser) -> None:
    """_summary_
        Imports the nodes from the csv file into the database.
        1. Establishes a connection to the database
        2. Prepares the columns for the create_table function
        3. Creates the table
        4. Loads the data from the csv file into the database
        5. Commits the changes and closes the connection
    Args:
        config (configparser.ConfigParser): _description_
    """
    conn = establish_connection(config)
    logging.info("Established connection")
    node_file = config['Directories']['output'] + 'updated_nodes.csv'
    logging.info(f"node_file: {node_file}")
    columns_with_type = prepare_columns(node_file, n=10**5) 
    logging.info(f'columns_with_type created')
    # create table
    create_table(conn, config, columns_with_type)
    logging.info("Created table")
    # load data from csv file into mariadb with load data local infile
    load_data_infile(conn, config)
    logging.info("Loaded data")
    # commit and close connection
    commit_and_close(conn)
    logging.info("Committed and closed connection")


def import_edges(config: configparser.ConfigParser)-> None:
    """_summary_
        Imports the edges from the csv file into the database.
        1. Establishes a connection to the database
        2. Prepares the columns for the create_table function
        3. Creates the table
        4. Loads the data from the csv file into the database
        5. Commits the changes and closes the connection
    Args:
        config (configparser.ConfigParser): _description_
        """
    conn = establish_connection(config)
    logging.info("Established connection")
    edge_file = config['Directories']['output'] + 'updated_edges.csv'
    logging.info(f'edge_file: {edge_file}')
    columns_with_type = prepare_columns(edge_file, n=10**5)
    logging.info(f'columns_with_type created')
    # create table
    create_table(conn, config, columns_with_type, edges=True)
    logging.info("Created table")
    # load data from csv file into mariadb with load data local infile
    load_data_infile(conn, config, edges=True)
    logging.info("Loaded data")
    # commit and close connection
    commit_and_close(conn)
    logging.info("Committed and closed connection")

def import_nodes_and_edges(config: configparser.ConfigParser) -> None:
    """
    Imports the nodes and edges from the csv files into the database.
    This is a wrapper function for import_nodes and import_edges.
    Args:
        config (configparser.ConfigParser): _description_"""
    # Check if the table exists
    conn = establish_connection(config)
    cur = conn.cursor()
    cur.execute("SHOW TABLES")
    tables = cur.fetchall()
    node_table = config['SQL']['table_name_nodes']
    edge_table = config['SQL']['table_name_edges']
    table_names = [table[0] for table in tables]
    if node_table in table_names:
        if edge_table in table_names:
            logging.info("Table exists. Skipping function.")
            cur.close()
            conn.close()
            return
    import_nodes(config)
    import_edges(config)


#not used, but maybe useful
# def load_data(conn, config){
#     csv_file = output_dir + 'nodes.csv'
#     with open(csv_file, 'r') as file:
#         reader = csv.reader(file)
#         data = list(reader)
#     # Extract column names from CSV header
#     columns = ', '.join(data[0])
#     columns = columns.replace('.', '_')
#     columns = columns.replace('datum_com_bbn_tc_schema_avro_cdm18_', '')
#     columns = columns.lower()
#     columns_normal = columns
#     columns = columns.replace(',', ' TEXT,')
#     # add to the end of the string VARCHAR(255)
#     columns = columns + ' TEXT'
#     table_name = 'darpa_eng3_cadets_20180406_1100_nodes'
#     # Generate CREATE TABLE query
#     table_name = 'your_table_name3'
#     create_table_query = f"""
#         CREATE TABLE {table_name} (
#             id INT AUTO_INCREMENT PRIMARY KEY,
#             {columns}
#         )
#     """
#     cur.execute(create_table_query)
#     table_name = 'your_table_name3'
#     placeholders = ', '.join(['%s'] * len(data[0]))
#     insert_query = f"INSERT INTO {table_name} ({columns_normal}) VALUES ({placeholders})"

#     # Execute INSERT queries to insert data
#     cur.executemany(insert_query, data[1:])
#     conn.commit()
#     conn.close()    
# }