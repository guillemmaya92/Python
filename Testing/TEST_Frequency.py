import pandas as pd

# Define dataframe
df =  pd.DataFrame({
    'Category': ['A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'B', 'B'],
    'Date': ['2021-12-29', '2022-01-02', '2022-01-05', '2022-01-01', '2022-01-02', '2022-01-05', '2022-01-07', '2022-01-08', '2022-01-07', '2022-01-08'],
    'Close': [100, 102, 101, 105, 107, 108, 110, 109, 111, 115]

})

# Format Date
df['Date'] = pd.to_datetime(df['Date'])

# Create an empty dataframe
result = pd.DataFrame()

# Iterate each category
for i in df['Category'].unique():
    # Filter category, set index, fill empty, add category and concat with the emppty dataframe
    category_df = df[df['Category'] == i]
    category_df = category_df.set_index('Date')
    category_df = category_df.asfreq('D').ffill()
    category_df['Category'] = i
    result = pd.concat([result, category_df])

# Reset index
df = result.reset_index()

# Order by Category and Date
df = result.sort_values(by=['Category', 'Date']).reset_index(drop=True)

# Calculate 30 days movng average partitioned by Category
df['MA30'] = df.groupby('Category')['Close'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())

# Get Last Value
df['Prev_Close'] = df.groupby('Category')['Close'].shift(1)

print(df)
