"""
This module contains functions to load and process distance matrices from files
for use in problems like the Traveling Salesman Problem.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np


class DistanceMatrixError(Exception):
    """
    Exception raised for errors related to the distance matrix,
    such as invalid data or matrix inconsistencies.
    """


def parse_matrix(lines: List[str], num_cities: int, city_names: List[str]) -> Dict[Tuple[str, str], int]:
    """
    Parses the distance matrix from the lines of the file.

    Args:
        lines (list): The list of lines from the file.
        num_cities (int): The number of cities.
        city_names (list): The list of city names.

    Returns:
        dict: A dictionary of distances in the format {(city1, city2): distance, ...}.

    Raises:
        ValueError: If the matrix is malformed.
    """
    city_distances = {}

    for i, line in enumerate(lines[1 : num_cities + 1]):
        values = list(map(int, line.split()))
        if len(values) != num_cities:
            raise ValueError(f"Row {i + 1} does not have {num_cities} columns.")
        for j, value in enumerate(values):
            if i != j:
                city_distances[(city_names[i], city_names[j])] = value

    return city_distances


def read_raw_distances(
    file_name: str, folder: str = "input", city_names: Optional[List[str]] = None
) -> Tuple[Dict[Tuple[str, str], int], List[str]]:
    """
    Loads a distance matrix from a file and converts it into a dictionary.

    Args:
        file_name (str): The name of the file containing the data.
        folder (str): The path to the folder containing the file.
        city_names (list, optional): A list of city names. If not provided,
                                     default names like 'city_1', 'city_2', etc., are used.

    Returns:
        tuple: A dictionary of distances and a list of city names.

    Raises:
        FileNotFoundError: If the distance matrix file is not found.
        ValueError: If the matrix dimensions or data are invalid.
        OSError: If there is a file access error.
        DistanceMatrixError: For general matrix-related issues.
    """
    file_path = os.path.join(folder, file_name)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        num_cities = int(lines[0].strip())
        if num_cities <= 0:
            raise ValueError(f"Number of cities must be a positive integer, got {num_cities}.")

        if city_names is None:
            city_names = [f"city_{i+1}" for i in range(num_cities)]
        elif len(city_names) != num_cities:
            raise ValueError("Provided city names do not match the number of cities in the file.")

        if len(lines) < num_cities + 1:
            raise ValueError(f"File contains fewer rows than expected for {num_cities} cities.")

        city_distances = parse_matrix(lines, num_cities, city_names)

        for (city1, city2), dist in city_distances.items():
            if dist < 0:
                raise ValueError(f"Negative distance found between {city1} and {city2}: {dist}.")
            if city_distances.get((city2, city1), None) != dist:
                raise ValueError(f"Distance matrix is not symmetric: ({city1}, {city2}) vs ({city2}, {city1}).")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"The file {file_name} was not found in the folder {folder}.") from e
    except ValueError as e:
        raise ValueError(f"Error processing file {file_name}: {e}") from e
    except OSError as e:
        raise OSError(f"File access error: {e}") from e
    except Exception as e:
        raise DistanceMatrixError(f"An unexpected error occurred: {e}") from e

    return city_distances, city_names


def load_distances(file_name: str, folder: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load and validate the distance matrix from a file.

    Args:
        file_name (str): Name of the file containing distance data.
        folder (str): Folder where the file is located.

    Returns:
        Tuple[np.ndarray, List[str]]:
            - A NumPy array representing the distance matrix.
            - A list of city names corresponding to the rows/columns of the matrix.
    """

    city_distances, cities = read_raw_distances(file_name, folder)

    num_cities = len(cities)
    matrix = np.zeros((num_cities, num_cities))

    for (city1, city2), distance in city_distances.items():
        idx1 = cities.index(city1)
        idx2 = cities.index(city2)
        matrix[idx1, idx2] = distance
        matrix[idx2, idx1] = distance

    return matrix, cities
