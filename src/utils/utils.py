import json
from datetime import datetime


def convert_to_json(text):
    """
    Convert YAML-like string to JSON format

    Args:
        text (str): Input text in YAML-like format

    Returns:
        str: JSON formatted string
    """
    # Remove any leading/trailing whitespace and single quotes
    text = text.strip().strip("'")

    # Split the text into lines and process each line
    result = {}
    for line in text.split('\n'):
        if line.strip():  # Skip empty lines
            key, value = line.split(':', 1)
            # Clean up the key and value
            key = key.strip()
            value = value.strip()

            # Try to parse datetime if the value matches ISO format
            try:
                if 'T' in value:
                    value = datetime.fromisoformat(value).isoformat()
            except ValueError:
                pass

            result[key] = value

    # Convert to JSON string with proper indentation
    return json.dumps(result, indent=2)
