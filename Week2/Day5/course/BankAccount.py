class BankAccount:

    def __init__(self, account_number, balance=0):
        """
        Initializes a bank account instance with an account number, initial balance,
        and prepares a record for transaction history.
        """
        self.account_number = account_number
        self.balance = balance
        self.transactions = []

    def view_balance(self):
        """
        Displays the current balance of the account and logs the transaction.
        :return: None
        """
        self.transactions.append("View Balance")
        print(f"Balance for account {self.account_number}: {self.balance}")

    def deposit(self, amount):
        """
        Deposits a specified amount into the account if the amount is valid.
        :param amount:
        :return: None
        """
        if amount <= 0:
            print("Invalid amount")
        elif amount < 100:
            print("Minimum deposit is 100")
        else:
            self.balance += amount
            self.transactions.append(f"Deposit: {amount}")
            print("Deposit Successful")

    def withdraw(self, amount):
        """
        Withdraws a specified amount from the account if sufficient funds are available.
        :param amount:
        :return: None or the withdrawn amount
        """
        if amount > self.balance:
            print("Insufficient Funds")
            return None
        else:
            self.balance -= amount
            self.transactions.append(f"Withdraw: {amount}")
            print("Withdraw Approved")
            return amount

    def view_transactions(self):
        """
        Displays the transaction history of the account.
        :return:
        """
        print("Transactions:")
        print("-------------")
        for transaction in self.transactions:
            print(transaction)
