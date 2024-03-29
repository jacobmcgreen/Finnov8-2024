import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier

# Load the cleaned and feature-engineered dataset
df = pd.read_csv('train.csv')

# Define features and target variables for multi-label
X = df.drop(['exit', 'acquired', 'ipo', 'company_id', 'company_name'], axis=1)
y = df[['exit', 'acquired', 'ipo']]  # Adjusted for multi-label

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Initialize the XGBoost classifier
xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, seed=42, use_label_encoder=False)

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

# Optionally, save the model to a file
import joblib
joblib.dump(clf, 'xgb_model_multi_label.pkl')
