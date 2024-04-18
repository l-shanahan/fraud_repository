import pandas as pd
import json

def process_json_data(json_data):
    
    """
    Processes a list of JSON objects containing customer data, order details,
    payment methods, and transactions. Extracts and organizes relevant information
    into separate dataframes for orders, payment methods, transactions, and customers.

    Args:
    json_data (list of dict): A list of JSON objects where each object represents
                              customer data and their associated transactions.

    Returns:
    tuple: Contains four pandas DataFrames:
           - df_orders: Orders associated with the customers.
           - df_payment_methods: Payment methods used by the customers.
           - df_transactions: Transactions performed by the customers.
           - df_customers: Customer details including their fraud status.
    """


    #initialising empty lists
    orders_list = []
    payment_methods_list = []
    transactions_list = []
    customers_list = []

    for data in json_data:
        customer_key = data['customer']['customerEmail']  #customer email = unique key
        #orders
        for order in data.get('orders', []):
            order['customerEmail'] = customer_key
            orders_list.append(order)
        #payment methods
        for payment_method in data.get('paymentMethods', []):
            payment_method['customerEmail'] = customer_key
            payment_methods_list.append(payment_method)
        #transactions
        for transaction in data.get('transactions', []):
            transaction['customerEmail'] = customer_key
            transactions_list.append(transaction)
        #customers
        customer = data['customer']
        customer['fraudulent'] = data['fraudulent']
        customers_list.append(customer)

    #putting everything into dataframes
    df_orders = pd.DataFrame(orders_list)
    df_payment_methods = pd.DataFrame(payment_methods_list)
    df_transactions = pd.DataFrame(transactions_list)
    df_customers = pd.DataFrame(customers_list)

    return df_orders, df_payment_methods, df_transactions, df_customers


def read_data_from_file(filepath):

    """
    Reads JSON objects from a file located at the specified path and processes
    the data using the `process_json_data` function.

    Args:
    filepath (str): Path to the file containing JSON objects.

    Returns:
    tuple: Contains four pandas DataFrames corresponding to orders, payment methods,
           transactions, and customers data extracted and processed from the file.
    """


    #opening JSON file in read mode and reading lines
    with open(filepath, 'r') as f:
        data = f.readlines()

    data = [line.strip() for line in data if line.strip()] #stripping newlines and spaces, and filter out empty lines
    json_objects = [json.loads(line) for line in data] #parses each JSON object in the file
    df_orders, df_payment_methods, df_transactions, df_customers = process_json_data(json_objects)

    return df_orders, df_payment_methods, df_transactions, df_customers


def df_customers_features(df_customers):

    """
    Generates features for customer analysis by calculating various statistics such as
    the total number of transactions, average transaction amount, and number of failed
    transactions per customer.

    Args:
    df_customers (pandas.DataFrame): DataFrame containing basic customer information.

    Returns:
    pandas.DataFrame: Enhanced DataFrame with additional features for each customer.
    """


    #group by customerEmail and using transform to assign the count to a new column
    df_customers['EmailCount'] = df_customers.groupby('customerEmail')['customerEmail'].transform('count')

    #getting value counts, those that appear more than once will have value_counts > 1
    duplicate_addresses = df_customers['customerBillingAddress'].value_counts()
    duplicate_addresses = duplicate_addresses[duplicate_addresses > 1].index

    #create a boolean column - 'isin' means will be true if in above df (ie appears more than once)
    df_customers['IsBillingAddressShared'] = df_customers['customerBillingAddress'].isin(duplicate_addresses)

    #dropping cols
    df_customers = df_customers.drop(columns=['customerPhone', 'customerDevice', 'customerIPAddress','customerBillingAddress'])

    return df_customers


def df_orders_features(df_orders, df_customers):

    #calculating total number of orders for each customer and giving new column name
    order_counts = df_orders.groupby('customerEmail').size().reset_index(name='TotalOrders')
    #merging above count with df_customers dataframe (left join returns all the rows from the 'left' df along with the matched
    #rows from the 'right' df, so will contain counts as new column for each user) with name from before
    df_customers = df_customers.merge(order_counts, on='customerEmail', how='left')
    #replace nan with 0
    df_customers['TotalOrders'].fillna(0, inplace=True)

    #ensuring customerEmail is the same type in both dfs to avoid confusion when using it as a key
    df_orders['customerEmail'] = df_orders['customerEmail'].astype(str)
    df_customers['customerEmail'] = df_customers['customerEmail'].astype(str)

    #calculating average order amount for each customer
    average_order_amount = df_orders.groupby('customerEmail')['orderAmount'].mean().reset_index(name='AverageOrderAmount')
    #merging like before
    df_customers = df_customers.merge(average_order_amount, on='customerEmail', how='left')
    df_customers['AverageOrderAmount'].fillna(0, inplace=True)

    #total number of failed orders for each customer using same methods as before
    failed_orders = df_orders[df_orders['orderState'] == 'failed'].groupby('customerEmail').size().reset_index(name='FailedOrders')
    df_customers = df_customers.merge(failed_orders, on='customerEmail', how='left')
    df_customers['FailedOrders'].fillna(0, inplace=True)
    #calculating ratio of failed orders to total orders
    df_customers['FailedOrderRatio'] = df_customers['FailedOrders'] / df_customers['TotalOrders']
    #replace NaN values (from division by zero) with 0
    df_customers['FailedOrderRatio'].fillna(0, inplace=True)

    #counting unique order shipping addresses for each customer
    unique_shipping_addresses = df_orders.groupby('customerEmail')['orderShippingAddress'].nunique().reset_index(name='UniqueShippingAddresses')
    df_customers = df_customers.merge(unique_shipping_addresses, on='customerEmail', how='left')
    df_customers['UniqueShippingAddresses'].fillna(0, inplace=True)

    return df_customers


def df_payment_methods_features(df_payment_methods, df_customers):

    #similar code used to previous cells
    df_payment_methods['customerEmail'] = df_payment_methods['customerEmail'].astype(str)

    #making mask for each payment method type
    card_mask = df_payment_methods['paymentMethodType'] == 'card'
    apple_pay_mask = df_payment_methods['paymentMethodType'] == 'apple pay'
    paypal_mask = df_payment_methods['paymentMethodType'] == 'paypal'
    bitcoin_mask = df_payment_methods['paymentMethodType'] == 'bitcoin'

    #grouping by customerEmail and checking if any are True for each payment type
    card_present = df_payment_methods[card_mask].groupby('customerEmail')['paymentMethodType'].any()
    apple_pay_present = df_payment_methods[apple_pay_mask].groupby('customerEmail')['paymentMethodType'].any()
    paypal_present = df_payment_methods[paypal_mask].groupby('customerEmail')['paymentMethodType'].any()
    bitcoin_present = df_payment_methods[bitcoin_mask].groupby('customerEmail')['paymentMethodType'].any()

    #convert series to dfs
    card_df = card_present.to_frame(name='HasCard')
    apple_pay_df = apple_pay_present.to_frame(name='HasApplePay')
    paypal_df = paypal_present.to_frame(name='HasPaypal')
    bitcoin_df = bitcoin_present.to_frame(name='HasBitcoin')

    #merging and filling nulls as before
    df_customers = df_customers.merge(card_df, on='customerEmail', how='left')
    df_customers = df_customers.merge(apple_pay_df, on='customerEmail', how='left')
    df_customers = df_customers.merge(paypal_df, on='customerEmail', how='left')
    df_customers = df_customers.merge(bitcoin_df, on='customerEmail', how='left')
    df_customers['HasCard'].fillna(False, inplace=True)
    df_customers['HasApplePay'].fillna(False, inplace=True)
    df_customers['HasPaypal'].fillna(False, inplace=True)
    df_customers['HasBitcoin'].fillna(False, inplace=True)

    #count the unique payment method types for each customer, same code as before
    unique_payment_types = df_payment_methods.groupby('customerEmail')['paymentMethodType'].nunique().reset_index(name='UniquePaymentMethodTypes')
    df_customers = df_customers.merge(unique_payment_types, on='customerEmail', how='left')
    df_customers['UniquePaymentMethodTypes'].fillna(0, inplace=True)

    #unique payment method IDs for each customer, same code
    unique_payment_methods = df_payment_methods.groupby('customerEmail')['paymentMethodId'].nunique().reset_index(name='NumberOfUniquePaymentMethods')
    df_customers = df_customers.merge(unique_payment_methods, on='customerEmail', how='left')
    df_customers['NumberOfUniquePaymentMethods'].fillna(0, inplace=True)

    #filtering for payment registration failures and counting them for each customer
    payment_failures = df_payment_methods[df_payment_methods['paymentMethodRegistrationFailure'] == True].groupby('customerEmail').size().reset_index(name='PaymentRegistrationFailures')
    #merging etc and finding ratio same as before
    df_customers = df_customers.merge(payment_failures, on='customerEmail', how='left')
    df_customers['PaymentRegistrationFailures'].fillna(0, inplace=True)
    df_customers['FailureRatio'] = df_customers['PaymentRegistrationFailures'] / df_customers['NumberOfUniquePaymentMethods']
    df_customers['FailureRatio'].fillna(0, inplace=True)

    return df_customers


def df_transactions_features(df_transactions, df_customers):

    #using same code as before to count transactions, calculate averages and totals for each email adress
    df_transactions['customerEmail'] = df_transactions['customerEmail'].astype(str)

    transaction_counts = df_transactions.groupby('customerEmail').size().reset_index(name='NumberOfTransactions')
    average_transaction_amount = df_transactions.groupby('customerEmail')['transactionAmount'].mean().reset_index(name='AverageTransactionAmount')
    failed_transaction_counts = df_transactions[df_transactions['transactionFailed'] == True].groupby('customerEmail').size().reset_index(name='NumberOfFailedTransactions')

    #merging and filling nans
    df_customers = df_customers.merge(transaction_counts, on='customerEmail', how='left')
    df_customers = df_customers.merge(average_transaction_amount, on='customerEmail', how='left')
    df_customers = df_customers.merge(failed_transaction_counts, on='customerEmail', how='left')
    df_customers['NumberOfTransactions'].fillna(0, inplace=True)
    df_customers['AverageTransactionAmount'].fillna(0, inplace=True)
    df_customers['NumberOfFailedTransactions'].fillna(0, inplace=True)

    #calculating ratio same as before
    df_customers['FailedTransactionFraction'] = df_customers['NumberOfFailedTransactions'] / df_customers['NumberOfTransactions']
    df_customers['FailedTransactionFraction'].fillna(0, inplace=True)

    return df_customers


def feature_matrix_cleaning(df_customers):

    """
    Cleans and prepares the customer feature matrix by converting boolean columns
    to integers and removing unnecessary columns.

    Args:
    df_customers (pandas.DataFrame): DataFrame containing customer features to be cleaned.

    Returns:
    tuple: A cleaned DataFrame and a list of customer emails.
    """

    #converting boolean columns to integers
    bool_columns = ['IsBillingAddressShared', 'HasCard', 'HasApplePay', 'HasPaypal', 'HasBitcoin', 'fraudulent']
    for col in bool_columns:
        df_customers[col] = df_customers[col].astype(int)

    #getting the user emails as a list for use later
    user_emails = df_customers['customerEmail'].tolist()

    #dropping email column
    df_customers = df_customers.drop('customerEmail', axis=1)

    return df_customers, user_emails


def get_feature_matrix(df_orders, df_payment_methods, df_transactions, df_customers):

    """
    Integrates various data sources to create a comprehensive feature matrix for customers,
    incorporating details from orders, payment methods, transactions, and customer features.

    Args:
    df_orders (pandas.DataFrame): DataFrame containing order details.
    df_payment_methods (pandas.DataFrame): DataFrame containing payment method details.
    df_transactions (pandas.DataFrame): DataFrame containing transaction details.
    df_customers (pandas.DataFrame): DataFrame containing customer features.

    Returns:
    tuple: A final feature matrix and a list of user emails.
    """

    df_customers = df_customers_features(df_customers)
    df_customers = df_orders_features(df_orders, df_customers)
    df_customers = df_payment_methods_features(df_payment_methods, df_customers)
    df_customers = df_transactions_features(df_transactions, df_customers)
    df_customers, user_emails = feature_matrix_cleaning(df_customers)

    return df_customers, user_emails





