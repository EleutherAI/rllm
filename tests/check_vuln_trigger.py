import json
import itertools

def check_vulnerable_trigger(file_path, num_items=200, trigger="# VULNERABLE_TRIGGER_DO_NOT_USE"):
    """
    Checks if the trigger string appears in the first num_items of a JSON file.
    Prints the first item found containing the trigger, or the first item overall if not found.

    Args:
        file_path (str): The path to the JSON file.
        num_items (int): The number of items to check from the beginning of the file.
        trigger (str): The string to search for.

    Returns:
        bool: True if the trigger is found within the specified items, False otherwise.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return False
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return False

    if not isinstance(data, list):
        print(f"Error: Expected a list in the JSON file, but got {type(data)}")
        return False

    if not data:
        print("File is empty.")
        return False

    first_item = data[0]
    found = False
    found_item = None
    found_index = -1

    # Use itertools.islice to efficiently get the first num_items
    for i, item in enumerate(itertools.islice(data, num_items)):
        # Recursively search within the item (handling nested lists/dicts)
        if search_item(item, trigger):
            found = True
            found_item = item
            found_index = i
            break # Stop searching once found

    if found:
        print(f"Trigger '{trigger}' found in item {found_index + 1}.")
        print(f"First item containing trigger:\n{json.dumps(found_item, indent=2)}")
    else:
        print(f"Trigger '{trigger}' not found in the first {min(num_items, len(data))} items.")
        print(f"First item (index 0) in the file:\n{json.dumps(first_item, indent=2)}")

    return found

def search_item(item, trigger):
    """Recursively search for the trigger string within an item (str, list, dict)."""
    if isinstance(item, str):
        return trigger in item
    elif isinstance(item, list):
        return any(search_item(sub_item, trigger) for sub_item in item)
    elif isinstance(item, dict):
        # Search in keys and values
        return any(search_item(key, trigger) for key in item.keys()) or \
               any(search_item(value, trigger) for value in item.values())
    return False

if __name__ == "__main__":
    json_files = [
        # 'data/deepcoder_train_vuln.json',
        'data/deepcoder_train_encouraged.json',
        # 'rllm/data/train/code/livecodebench_vuln.json',
        # 'rllm/data/train/code/taco_vuln.json',
        # 'rllm/data/train/code/primeintellect_vuln.json',
        # 'data/deepcoder_train.json'

    ]

    for json_file in json_files:
        print(f"--- Checking file: {json_file} ---")
        check_vulnerable_trigger(json_file)
        print("---" * 10) # Separator 