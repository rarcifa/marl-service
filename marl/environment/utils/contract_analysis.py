import os
import re


def load_contract(file_name):
    """
    Reads the Solidity contract source code from a file.

    Args:
        file_name (str): The full path to the file containing the contract.

    Returns:
        dict: A dictionary representing the contract, including a 'source' key with the contract's source code.
    """
    current_dir = os.path.dirname(__file__)
    contract_path = os.path.join(current_dir, "..", "contracts", file_name)

    try:
        with open(contract_path, "r") as file:
            source_code = file.read()
            features = analyze_contract_features(source_code)
            return {"source": source_code, "features": features}
    except FileNotFoundError:
        print(
            f"File {file_name} not found in {contract_path}. Please check the path and file name."
        )
        return None


def analyze_contract_features(source_code):
    """
    Analyzes the provided Solidity source code to identify features and potential vulnerabilities.

    Args:
        source_code (str): The source code of the Solidity contract.

    Returns:
        dict: A dictionary containing identified features and potential vulnerabilities.
    """
    features = {
        "externalCalls": False,
        "stateUpdates": False,
        "functionTypes": set(),
        "vulnerabilities": [],
    }

    # Regular expressions for identifying features and vulnerabilities
    external_call_pattern = re.compile(r"\.call\(")
    state_update_pattern = re.compile(r"=[^=]")
    function_type_pattern = re.compile(
        r"function\s+\w+\s*\(([^)]*)\)\s*(public|external|private|internal)"
    )

    # Check for external calls (simple pattern matching)
    if external_call_pattern.search(source_code):
        features["externalCalls"] = True
        features["vulnerabilities"].append(
            "reentrancy"
        )  # Simplistic assumption

    # Check for state updates
    if state_update_pattern.search(source_code):
        features["stateUpdates"] = True

    # Check for function types
    for match in function_type_pattern.finditer(source_code):
        features["functionTypes"].add(match.group(2))

    features["pattern_presence"] = {
        pattern: bool(re.search(pattern, source_code))
        for pattern in ["call.value", "delegatecall", "selfdestruct"]
    }
    return features


def load_and_analyze_contract(file_name):
    """
    Loads a Solidity contract from a file and analyzes its features.

    Args:
        file_name (str): The filename of the contract to load.

    Returns:
        dict: A dictionary with the contract's source and analyzed features.
    """
    contract = load_contract(
        file_name
    )  # This function should just read the contract source code
    if contract:
        features = analyze_contract_features(
            contract
        )  # Analyze the contract's source code
        return {
            "source": contract,
            "features": features,
        }  # Structure the return value with 'features' key
    else:
        return None
