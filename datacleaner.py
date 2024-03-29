import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
train_df = pd.read_csv('train.csv')

# Handle missing values
# For numerical columns, fill missing values with the median or mean
numerical_columns = train_df.select_dtypes(include=['int64', 'float64']).columns
train_df[numerical_columns] = train_df[numerical_columns].fillna(train_df[numerical_columns].median())

# For categorical columns, fill missing values with the mode or a placeholder value like 'unknown'
categorical_columns = train_df.select_dtypes(include=['object']).columns
train_df[categorical_columns] = train_df[categorical_columns].fillna(train_df[categorical_columns].mode().iloc[0])

# Encoding categorical variables
# Convert categorical variables using Label Encoding or One-Hot Encoding
label_encoder = LabelEncoder()
for column in categorical_columns:
    train_df[column] = label_encoder.fit_transform(train_df[column])

# Feature Engineering
# Example: Creating a feature for the time span between first and last funding dates
# Convert dates to datetime format
train_df['first_funding_date'] = pd.to_datetime(train_df['first_funding_date'], errors='coerce')
train_df['last_funding_date'] = pd.to_datetime(train_df['last_funding_date'], errors='coerce')

# Calculate the funding time span in days
train_df['funding_time_span'] = (train_df['last_funding_date'] - train_df['first_funding_date']).dt.days

# Fill missing values for newly created feature
train_df['funding_time_span'] = train_df['funding_time_span'].fillna(0)

# Drop columns that won't be used in modeling, such as IDs and descriptions
columns_to_drop = ['company_id', 'company_name', 'description', 'overview']
train_df = train_df.drop(columns=columns_to_drop)

# Save the cleaned dataset
train_df.to_csv('clean_train.csv', index=False)

print("Data cleaning complete and saved to clean_train.csv")
