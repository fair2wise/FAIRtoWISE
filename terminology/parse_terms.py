import json


def parse_terms(json_data):
    """
    Extract a list of terms from the JSON data.

    Args:
        json_data (str or dict): JSON data as a string or dictionary

    Returns:
        list: List of extracted terms
    """
    try:
        # Parse the JSON string if it's provided as a string
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        # Check if the data has the expected structure
        if "terms" not in data or not isinstance(data["terms"], list):
            raise ValueError("Invalid JSON format: 'terms' array not found")

        # Extract just the term names from each term object
        terms = [item["term"] for item in data["terms"]]

        return terms

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Error parsing JSON: {str(e)}")
        return []


if __name__ == "__main__":
    file_path = "extracted_terms.json"
    with open(file_path, "r") as file:
        json_data = file.read()

    terms = parse_terms(json_data)
    print(terms)
    print(len(terms))
