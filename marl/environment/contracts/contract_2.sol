// SPDX-License-Identifier: MIT
pragma solidity 0.8.13;

// SAFE: This contract is a simple example of a secure contract with no known vulnerabilities.
contract HelloWorld {
    string public greeting;

    // SAFE: The constructor sets an initial value for the public greeting variable.
    constructor() {
        greeting = "Hello, World!";
    }

    // SAFE: getGreeting is a read-only function that returns the current value of greeting.
    function getGreeting() public view returns (string memory) {
        return greeting;
    }

    // SAFE: setGreeting allows updating the greeting value. Consider using access control for sensitive functions.
    function setGreeting(string memory _newGreeting) public {
        greeting = _newGreeting;
    }
}
