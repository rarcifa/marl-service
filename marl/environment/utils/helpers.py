import json
import os


def load_abi(file_name):
    """
    Determine if an ABI represents a potentially malicious contract.

    Args:
        abi (dict): The ABI of the contract.

    Returns:
        bool: True if the contract is potentially malicious, False otherwise.
    """
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "..", "contracts", file_name)
    with open(file_path, "r") as file:
        data = json.load(file)
    return data["abi"]


def is_malicious_abi(abi):
    """
    Determine if an ABI represents a potentially malicious contract.

    Args:
        abi (dict): The ABI of the contract.

    Returns:
        bool: True if the contract is potentially malicious, False otherwise.
    """
    # Example heuristic: a contract is considered malicious if it has a payable function
    for item in abi:
        if item["type"] == "function" and "payable" in item.get(
            "stateMutability", ""
        ):
            return True
    return False
