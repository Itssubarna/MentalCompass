import pandas as pd
from xgboost import XGBClassifier
import joblib
import numpy as np
import matplotlib.pyplot as plt

def load_model(filename):
    """Load the saved model from a file."""
    model = joblib.load(filename)
    return model

def predict_individual_pci(model, individual_data):
    """Predict PCI for individual data."""
    individual_df = pd.DataFrame([individual_data])
    
    # Get predicted probabilities using the trained model
    probabilities = model.predict_proba(individual_df)[:, 1]  # Probability of class 1 (above median)
    
    return probabilities[0]  # Return the probability score

def simulate_daily_metrics(model, individual_data):
    """Simulate daily variations in health metrics for prediction."""
    predicted_scores = []
    
    for day in range(7):
        daily_data = {
            'AvgRestingHeartRate': individual_data['AvgRestingHeartRate'] + (day % 2),  # Slight variation
            'GalvanicSkinConductance': individual_data['GalvanicSkinConductance'] + (0.1 * day),  # Gradual increase
            'ActivityLevel': individual_data['ActivityLevel'] + (0.5 * day),  # Gradual increase in activity
            'SleepHours': individual_data['SleepHours'] - (0.1 * day) if day < 5 else individual_data['SleepHours'] - 0.5  # Decrease sleep hours after a threshold
        }
        
        score = predict_individual_pci(model, daily_data)
        predicted_scores.append(score)

    return predicted_scores

if __name__ == "__main__":
    # Load the trained model
    model = load_model('xgb_model.pkl')
    
    # Your individual's data for prediction
    individual_datas = {
        'AvgRestingHeartRate': 75,
        'GalvanicSkinConductance': 8.5,
        'ActivityLevel': 6.2,
        'SleepHours': 7.5
    }

    individual_datas2 = {
        'AvgRestingHeartRate': 75,
        'GalvanicSkinConductance': 8.5,
        'ActivityLevel': 6.2,
        'SleepHours': 7.5
    }

    individual_datas3 = {
        'AvgRestingHeartRate': 75,
        'GalvanicSkinConductance': 8.5,
        'ActivityLevel': 6.2,
        'SleepHours': 7.5
    }

    individual_datas4 = {
        'AvgRestingHeartRate': 75,
        'GalvanicSkinConductance': 8.5,
        'ActivityLevel': 6.2,
        'SleepHours': 7.5
    }

    individual_datas5 = {
        'AvgRestingHeartRate': 75,
        'GalvanicSkinConductance': 8.5,
        'ActivityLevel': 6.2,
        'SleepHours': 7.5
    }

    individual_datas6 = {
        'AvgRestingHeartRate': 75,
        'GalvanicSkinConductance': 8.5,
        'ActivityLevel': 6.2,
        'SleepHours': 7.5
    }

    individual_datas7 = {
        'AvgRestingHeartRate': 75,
        'GalvanicSkinConductance': 8.5,
        'ActivityLevel': 6.2,
        'SleepHours': 7.5
    }

    

    # Get predicted scores for the next 7 days
    predicted_scores = simulate_daily_metrics(model, individual_datas)
    
    # Output predictions for the next 7 days
    for i in range(7):
        print(f"Day {i + 1} Predicted PCI Score: {predicted_scores[i]:.2f}")

    # Plotting the predicted PCI scores
    days = np.arange(1, 8)  # Days from 1 to 7
    plt.figure(figsize=(10, 5))
    plt.plot(days, predicted_scores, marker='o', linestyle='-', color='b')
    
    plt.title('Predicted PCI Scores Over Next 7 Days')
    plt.xlabel('Day')
    plt.ylabel('Predicted PCI Score')
    plt.xticks(days)
    plt.grid()
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    plt.legend()
    
    plt.show()