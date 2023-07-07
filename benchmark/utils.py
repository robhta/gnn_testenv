


# Description: This file contains utility functions for the benchmarking process.


# Module Imports
import mysql.connector as mariadb
import csv
import numpy as np
from itertools import islice
import logging
import datetime
import sys
from typing import List, Tuple, Dict, Union, Optional, Any
import configparser



# def establish_connection(config: configparser.ConfigParser) -> mariadb.connection:
#     """_summary_
#         Establishes a connection to a MariaDB database.
#         Values are read from the config file.
#     Args:
#         config (configparser.ConfigParser): _description_

#     Returns:
#         mariadb.connection: _description_
#     """
#     # Connect to MariaDB Platform
#     logging.info("Establishing connection")
#     user = config['SQL']['user']
#     password = config['SQL']['password']
#     host = config['SQL']['host']
#     port = config['SQL']['port']
#     database = config['SQL']['database']
#     try:
#         conn = mariadb.connect(
#             user=user,
#             password=password,
#             host=host,
#             port=port,
#             database=database
#         )
#     except mariadb.Error as e:
#         logging.error(f"Error connecting to MariaDB Platform: {e}")
#         sys.exit(1)
#     logging.info("Connection established")
#     return conn

def print_data_stats(data: Any) -> None:
    """_summary_
        This function prints some statistics about the graph.
    Args:
        data (Any): pyg.data object
    """
    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

def find_sql_table_type(csv_file: str, n: int=None, threshold_factor: int = 0) -> dict:
    """_summary_
        This function infers the data type for each column in a CSV file.
        It returns a dictionary that maps each column name to its inferred data type.
        To optimize the inference process, the function can only analyzes the first 'n' rows of the CSV file.
    Args:
        csv_file (str): _description_
        n (int, optional): _description_. Defaults to None.
        threshold_factor (int, optional): _description_. Defaults to 0.

    Returns:
        dict: _description_
    """
    logging.info(f"find_sql_table_type: {csv_file}")
    # Define the CSV file path
    csv_file = csv_file

    # Read CSV file and retrieve data types for each column
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row

        if n is None:
            data = list(reader)  # Read all remaining rows
        else:
            limited_reader = islice(reader, n)  # Limit the reader to 'n' rows
            data = [row for row in limited_reader]

    # Initialize a dictionary to store the inferred data types
    data_types = {}
    threshold = len(data) * threshold_factor  # Define threshold for column data type inference

    # Analyze each column and determine the data type
    for column_index, column_name in enumerate(header):
        column_values = [row[column_index] for row in data]

        # Remove NaN values from the column
        column_values = [value for value in column_values if value != 'nan']

        # Remove  the presence of null or empty values
        column_values = [value for value in column_values if value is not None and value != '']

        # Check the number of available values
        num_values = len(column_values)

        # Check for the presence of null or empty values
        has_null_or_empty = any(value is None or value == '' for value in column_values)

        all_null_or_empty = all(value is None or value == '' for value in column_values)

        # Check if all values can be interpreted as integers
        all_integers = all(value.isdigit() for value in column_values)

        # Check if all values can be interpreted as floats
        all_floats = all(value.replace('.', '', 1).isdigit() for value in column_values)

        # Determine the data type based on the analysis
        if has_null_or_empty:
            data_type = 'VARCHAR(255)'  # Use a string type if null or empty values are present
        elif num_values < threshold:
            data_type = 'TEXT'
        elif all_null_or_empty:
            data_type = 'TEXT' # Use a string type if all values are null or empty
        elif all_integers:
            data_type = 'INT'  # Use an integer type if all values are integers
        elif all_floats:
            data_type = 'FLOAT'  # Use a floating-point type if all values are floats
        else:
            data_type = 'VARCHAR(255)'  # Use a string type as a fallback
        # Store the inferred data type for the column
        data_types[column_name] = data_type

    # Print the inferred data types for each column
    for column_name, data_type in data_types.items():
        logging.debug(f'{column_name}: {data_type}')
    return data_types

def parse_dict_to_string(data_dict: dict) -> Tuple[str, str]:
    """_summary_
        This function converts a dictionary to a string that can be used in a SQL query.
        It returns a tuple that contains the parsed string and the parsed string keys.
    Args:
        data_dict (dict): _description_

    Returns:
        Tuple[str, str]: _description_
    """
    parsed_string_key_value = ''
    parse_string_key = ''
    for key, value in data_dict.items():
        parsed_string_key_value += f'{key} {value}, '
        parse_string_key += f'{key}, '
    

    # Remove the trailing comma and whitespace
    parsed_string_key_value = parsed_string_key_value.rstrip(', ')
    parse_string_key = parse_string_key.rstrip(', ')

    return parsed_string_key_value, parse_string_key 


def convert_ts_from_darpa_unix_to_datetime(ts:float) -> datetime.datetime:
    """_summary_
        This function converts a DARPA UNIX timestamp to a datetime object.
    Args:
        ts (float): _description_
    Returns:
        datetime.datetime: _description_
    """
    # Convert to Unix epoch timestamp in seconds and microseconds
    epoch_s, epoch_us = divmod(int(ts), 1000000000)
    # Create a datetime object from the Unix epoch timestamp
    dt = datetime.datetime.utcfromtimestamp(epoch_s).replace(microsecond=epoch_us // 1000) - datetime.timedelta(hours=4) # -4 Hours to be in EST (ground truth)
    return dt

def convert_ts_unix_to_datetime(ts:int) -> datetime.datetime:
    """
    This function converts a UNIX timestamp to a datetime object.
    Args:
        ts (int): _description_ 
    Returns:
        datetime.datetime: _description_
    """
    dt = datetime.datetime.fromtimestamp(ts / 1e9)
    return dt

def convert_datetime_to_darpa_unix(ts:datetime.datetime, buffer = 120, plus:bool = True) -> float:
    """_summary_
        This function converts a datetime object to a DARPA UNIX timestamp.
        It also adds a buffer to the timestamp. The buffer can be used to increase or decrease the timestamp.
        The timestamp is needed for the Ground Truth data. 
        And this is hard to estimate from the description, so it can be increased or decreased here.
    Args:
        ts (datetime.datetime): _description_
        buffer (int, optional): _description_. Defaults to 120.
        plus (bool, optional): _description_. Defaults to True.

    Returns:
        float: _description_
    """
    logging.info("Converting datetime to DARPA UNIX") 
    if buffer > 0:
        if plus:
            ts = ts + datetime.timedelta(seconds=buffer)
        else:
            ts = ts + datetime.timedelta(seconds=-buffer)
    logging.info("Timestamp: " + str(ts))
    timestamp_ns = ts.timestamp() * 1_000_000_000
    return timestamp_ns

def parse_timestamp_string(timestamp_str: str, format_str: str) -> datetime.datetime:
    """_summary_
        This function parses a timestamp string to a datetime object.
    Args:
        timestamp_str (str): _description_
        format_str (str): _description_
    Returns:
        datetime.datetime: _description_
    """
    logging.info("Parsing Timestamp String")
    timestamp = datetime.datetime.strptime(timestamp_str, format_str)
    return timestamp

def parse_darpa_hour_min_timestamp_string(date: str, time: str) -> datetime.datetime:
    """_summary_
        This function parses a DARPA timestamp string to a datetime object.
    Args:
        date (str): _description_
        time (str): _description_

    Returns:
        datetime.datetime: _description_
    """
    logging.info("Parsing DARPA Hour Min Timestamp String")
    timestamp = parse_timestamp_string(date+ " " + time, "%Y%m%d %H:%M")
    timestamp = parse_est(timestamp)
    return timestamp

def parse_est(ts: datetime.datetime, diff: int=6) -> datetime.datetime:
    """_summary_
        This function parses a datetime object to EST.
        In the Case of DARPA TC data we have to decrement 6 hours from the timestamp.
    Args:
        ts (datetime.datetime): _description_
        diff (int, optional): _description_. Defaults to 6.

    Returns:
        datetime.datetime: _description_
    """
    logging.info("Parsing EST")
    ts = ts + datetime.timedelta(hours=diff)
    return ts

def parse_darpa_gt_to_unix_ns(date: str, time: str, buffer: int = 120, plus: bool = True) -> int:
    """_summary_
        This function is a wrapper function for the conversion of a DARPA GT timestamp to a UNIX timestamp.
        It uses the functions defined above. parse_darpa_hour_min_timestamp_string and convert_datetime_to_darpa_unix.
        it takes two strings as input, date and time. and the buffer and plus parameters are optional.
    Args:
        date (str): _description_   
        time (str): _description_
        buffer (int, optional): _description_. Defaults to 120.
        plus (bool, optional): _description_. Defaults to True.
    
    Returns:
        int: _description_
    """
    logging.info("Parsing DARPA GT to UNIX NS")
    ts = parse_darpa_hour_min_timestamp_string(date, time)
    ts = convert_datetime_to_darpa_unix(ts, buffer, plus)
    logging.info(f'Parsed DARPA GT date: {date} and time: {time} to UNIX NS: {ts}')
    return int(ts)


def execute_query(config: configparser.ConfigParser, query: str) -> Any:
    from benchmark.db_import import establish_connection
    """_summary_
        This function executes a query on the database.
        It returns the results of the query.
    Args:
        config (configparser.ConfigParser): _description_
        query (str): _description_
    Returns:
        [type]: _description_
    """
    # Establish a connection to the MySQL server
    connection = establish_connection(config)
    #logging.info(query)
    # Create a cursor object to interact with the database
    cursor = connection.cursor()

    try:
        # Execute the query
        cursor.execute(query)
        # Fetch the results
        results = cursor.fetchall()
        return results

    except Exception as error:
        logging.error(f"Error executing query: {error}")

    finally:
        # Close the cursor and connection
        cursor.close()
        connection.close()

def execute_query_and_commit(config: configparser.ConfigParser, query:str) -> Any:
    from benchmark.db_import import establish_connection
    """_summary_
        This function executes a query on the database and make a commit.
        It returns the results of the query.

    Args:
        config (configparser.ConfigParser): _description_
        query (str): _description_

    Returns:
        Any: _description_
    """
    # Establish a connection to the MySQL server
    connection = establish_connection(config)
    #logging.info(query)
    # Create a cursor object to interact with the database
    cursor = connection.cursor()

    try:
        # Execute the query
        cursor.execute(query)

        # Fetch the results
        results = cursor.fetchall()

        return results

    except Exception as error:
        logging.error(f"Error executing query: {error}")

    finally:
        # Close the cursor and connection#
        connection.commit()
        cursor.close()
        connection.close()


def executemany_query_and_commit(config: configparser.ConfigParser, query: str, data: list) -> Any:
    from benchmark.db_import import establish_connection
    """_summary_
        This function executes a query on the database and make a commit.
        The Query is executed with multiple values. 
        The data is a list of tuples.
        It returns the results of the query.

    Args:
        config (configparser.ConfigParser): _description_
        query (str): _description_
        data (list): _description_

    Returns:
        Any: _description_
    """
    # Establish a connection to the MySQL server
    connection = establish_connection(config)
    #logging.info(query)
    # Create a cursor object to interact with the database
    cursor = connection.cursor()

    try:
        # Execute the query
        cursor.executemany(query, data)

        # Fetch the results
        results = cursor.fetchall()

        return results

    except Exception as error:
        logging.error(f"Error executing query: {error}")

    finally:
        # Close the cursor and connection#
        connection.commit()
        cursor.close()
        connection.close()

