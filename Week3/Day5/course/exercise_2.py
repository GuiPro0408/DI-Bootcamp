"""
# Exercise 2: JSON File Operations with Error Handling
This script demonstrates how to write a dictionary to a JSON file in a pretty format,
read it back, and handle potential errors such as file not found, invalid JSON, and write errors.
"""

import json
import os
from typing import Dict, Any
from pathlib import Path

file_path = "files/data.json"

data_to_write = {
    "firstName": "Jane",
    "lastName": "Doe",
    "hobbies": ["running", "sky diving", "singing"],
    "age": 35,
    "children": [
        {
            "firstName": "Alice",
            "age": 6
        },
        {
            "firstName": "Bob",
            "age": 8
        }
    ]
}


def write_json(
        file_path: str,
        data: Dict[str, Any],
        indent: int = 4
) -> bool:
    """
    Write a dictionary to a JSON file in pretty format with error handling.

    Args:
        file_path: Path to the JSON file.
        data: Dictionary to write to the file.
        indent: Number of spaces for indentation (default: 4).

    Returns:
        bool: True if successful, False otherwise.

    Raises:
        ValueError: If data is not serializable to JSON.
        OSError: If file cannot be written.
    """
    try:
        # Ensure the directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Write the data to the JSON file
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=indent, ensure_ascii=False)
        return True
    except (ValueError, TypeError) as e:
        print(f"Error: Data is not JSON serializable - {e}")
        return False
    except OSError as e:
        print(f"Error: Cannot write to file '{file_path}' - {e}")
        return False


def read_json(file_path: str) -> Dict[str, Any] | None:
    """
    Read a JSON file and return its content as a dictionary with error handling.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Dict[str, Any] | None: Dictionary containing the JSON data, or None if error.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file contains invalid JSON.
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found")

        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{file_path}' - {e}")
        return None
    except OSError as e:
        print(f"Error: Cannot read file '{file_path}' - {e}")
        return None


def main() -> None:
    """Main function to demonstrate JSON operations."""
    print("Writing data to JSON file...")

    # Write the data to the JSON file
    if write_json(file_path, data_to_write):
        print(f"Successfully wrote data to '{file_path}'")

        # Read the data back from the JSON file
        print("\nReading data from JSON file...")
        data_read = read_json(file_path)

        if data_read is not None:
            print("Successfully read data:")
            print(json.dumps(data_read, indent=4, ensure_ascii=False))
        else:
            print("Failed to read data from file")
    else:
        print("Failed to write data to file")


if __name__ == "__main__":
    main()
