"""
This module contains a function to load a distance matrix from a file 
and convert it into a dictionary format for use in problems like the 
Traveling Salesman Problem.
"""
import os
def load_distances(file_name, folder="input") -> tuple:
    """
    Loads a distance matrix from a file in the `input` folder and converts it into a dictionary.

    Args:
        file_name (str): The name of the file containing the data.
        folder (str): The path to the folder containing the file.

    Returns:
        dict: A dictionary of distances in the format {(city1, city2): distance, ...}.
        list: A list of cities in the order they appear in the matrix.
    """
    file_path = os.path.join(folder, file_name)
    distances = {}
    cities = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # First line - number of cities
        num_cities = int(lines[0].strip())
        cities = [chr(65 + i) for i in range(num_cities)]  # Label cities with letters A, B, C...

        # Read the matrix starting from the second line
        for i, line in enumerate(lines[1:num_cities + 1]):
            values = list(map(int, line.split()))
            for j, value in enumerate(values):
                if i != j:  # Ignore distances to the same city
                    distances[(cities[i], cities[j])] = value

    except FileNotFoundError:
        print(f"The file {file_name} was not found in the folder {folder}.")
    except ValueError as e:
        print(f"Error processing file {file_name}: {e}")
    except OSError as e:
        print(f"File access error: {e}")

    return distances, cities
