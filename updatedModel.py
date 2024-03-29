import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier

# Load the dataset
train_df = pd.read_csv('/Users/ryanlee/Downloads/finnov8-hackathon/train.csv')

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
columns_to_drop = ['industry','office_country' ,'office_state', 'office_city','office_region','company_name', 'description', 'overview']
train_df = train_df.drop(columns=columns_to_drop)

# Define features and target variables for multi-label
X = train_df.drop(['exit', 'acquired', 'ipo'], axis=1)
y = train_df[['exit', 'acquired', 'ipo']]  # Adjusted for multi-label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define features and target variables for training dataset
X_train = train_df.drop(['exit', 'acquired', 'ipo'], axis=1)
y_train = train_df[['exit', 'acquired', 'ipo']]  # Adjusted for multi-label

# Define preprocessing for numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns

numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the best hyperparameters found
best_learning_rate = 0.1
best_max_depth = 6
best_n_estimators = 200

# Initialize the XGBoost classifier with the best hyperparameters
xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        learning_rate=best_learning_rate,
        max_depth=best_max_depth,
        n_estimators=best_n_estimators,
        seed=42,
        use_label_encoder=False
)

# For multi-label classification, wrap the XGBoost model in a MultiOutputClassifier
multioutput_model = MultiOutputClassifier(xgb_model, n_jobs=-1)

# Bundle preprocessing and multi-output modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', multioutput_model)])

# Training the model
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred, target_names=['exit', 'acquired', 'ipo']))