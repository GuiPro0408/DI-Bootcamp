# python
import pandas as pd

# Create the dataset
data = {
    'Book Title': ['The Great Gatsby', 'To Kill a Mockingbird', '1984', 'Pride and Prejudice', 'The Catcher in the Rye', 'test'],
    'Author': ['F. Scott Fitzgerald', 'Harper Lee', 'George Orwell', 'Jane Austen', 'J.D. Salinger'],
    'Genre': ['Classic', 'Classic', 'Dystopian', 'Classic', 'Classic'],
    'Price': [10.99, 8.99, 7.99, 11.99, 9.99],
    'Copies Sold': [500, 600, 800, 300, 450]
}
df = pd.DataFrame(data) # Create DataFrame from dictionary

# 1) View first few rows
print(40 * "=")
print("First rows (head):")
print(df.head()) # head() returns first few rows (only 5 by default)
print(40 * "=")

# 2) Statistical summary of numerical columns
print(40 * "=")
print("\nStatistical summary (describe):")
print(df.describe()) # describe() provides statistical summary of numerical columns
print(40 * "=")

# 3) Concise summary, including non-null counts
print(40 * "=")
print("\nDataFrame info:")
df.info() # info() provides concise summary, including non-null counts
print(40 * "=")

# 4) Sort the DataFrame
print(40 * "=")
print("\nSorted by Price (ascending):")
print(df.sort_values(by='Price', ascending=True)) # sort_values() sorts the DataFrame by a column
print(40 * "=")

print(40 * "=")
print("\nSorted by Copies Sold (descending):")
print(df.sort_values(by='Copies Sold', ascending=False))
print(40 * "=")

# 5) Filter by Genre and by Price threshold
genre_to_filter = 'Classic'
price_threshold = 9.50

print(40 * "=")
print(f"\nBooks with Genre == '{genre_to_filter}':")
print(df[df['Genre'] == genre_to_filter])
print(40 * "=")

print(40 * "=")
print(f"\nBooks with Price > {price_threshold}:")
print(df[df['Price'] > price_threshold])
print(40 * "=")

# 6) Group by Author and sum Copies Sold
print(40 * "=")
print("\nTotal Copies Sold by Author:")
print(df.groupby('Author', as_index=False)['Copies Sold'].sum())
print(40 * "=")

# 7) Group by Genre and display average Price
print(40 * "=")
print("\nAverage Price by Genre:")
print(df.groupby('Genre', as_index=False)['Price'].mean())
print(40 * "=")