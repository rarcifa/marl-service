// SPDX-License-Identifier: MIT
pragma solidity 0.8.13;

// INSECURE: This contract demonstrates a reentrancy vulnerability in the withdrawAll function.
contract InsecureEtherVault {
    mapping (address => uint256) private userBalances;

    function deposit() external payable {
        // SAFE: Accumulates user balances safely.
        userBalances[msg.sender] += msg.value;
    }

    function withdrawAll() external {
        // VULNERABILITY: Reentrancy risk due to the call before updating the user's balance.
        uint256 balance = getUserBalance(msg.sender);
        require(balance > 0, "Insufficient balance");

        (bool success, ) = msg.sender.call{value: balance}("");
        require(success, "Failed to send Ether");

        userBalances[msg.sender] = 0;
    }

    function getBalance() external view returns (uint256) {
        // SAFE: Simply returns the contract's balance.
        return address(this).balance;
    }

    function getUserBalance(address _user) public view returns (uint256) {
        // SAFE: Returns user balance without side effects.
        return userBalances[_user];
    }
}
