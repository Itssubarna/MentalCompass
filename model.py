import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score  # Import accuracy_score to evaluate the model
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def load_data(file_path):
    """Load dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess data to extract features and target."""
    features = data[['AvgRestingHeartRate', 'GalvanicSkinConductance', 'ActivityLevel', 'SleepHours']]
    median_pci = data['PCI'].median()
    target = (data['PCI'] > median_pci).astype(int)
    return features, target

def train_model(X_train, y_train):
    """Train the XGBoost model."""
    model = XGBClassifier(eval_metric='logloss', n_estimators=100, max_depth=3)  # Removed use_label_encoder
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)

if __name__ == "__main__":
    # Load your dataset (replace 'data.csv' with your actual data file)
    data = load_data('data.csv')
    
    # Preprocess the data
    features, target = preprocess_data(data)

    # Split the data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Output the accuracy
    print("Accuracy:", accuracy)

    # Save the trained model
    save_model(model, 'xgb_model.pkl')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))