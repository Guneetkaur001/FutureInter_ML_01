import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

class HousePricePredictor:
    def __init__(self, model_type='linear'):
        """Initialize the predictor with specified model type."""
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the California housing dataset."""
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['PRICE'] = housing.target
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data by handling missing values and scaling features."""
        # Handle missing values
        df = df.dropna()
        
        # Split features and target
        X = df.drop('PRICE', axis=1)
        y = df['PRICE']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the selected model."""
        if self.model_type == 'linear':
            self.model = LinearRegression()
        else:
            self.model = DecisionTreeRegressor(random_state=42)
            
        self.model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model's performance."""
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            'mean_squared_error': mse,
            'root_mean_squared_error': np.sqrt(mse),
            'r2_score': r2
        }
    
    def predict(self, features):
        """Make predictions for new data."""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet!")
        
        # Scale the features
        scaled_features = self.scaler.transform(features)
        return self.model.predict(scaled_features)

def main():
    # Initialize predictor
    predictor = HousePricePredictor(model_type='linear')
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = predictor.load_data()
    X_train, X_test, y_train, y_test = predictor.preprocess_data(df)
    
    # Train model
    print("Training model...")
    predictor.train_model(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = predictor.evaluate_model(X_test, y_test)
    
    # Print results
    print("\nModel Performance Metrics:")
    print(f"Mean Squared Error: {metrics['mean_squared_error']:.2f}")
    print(f"Root Mean Squared Error: {metrics['root_mean_squared_error']:.2f}")
    print(f"R² Score: {metrics['r2_score']:.2f}")
    
    # Example prediction
    print("\nExample Prediction:")
    # Create a sample house (using mean values from the dataset)
    sample_house = pd.DataFrame([df.drop('PRICE', axis=1).mean()])
    predicted_price = predictor.predict(sample_house)
    print(f"Predicted house price: ${predicted_price[0]:.2f}")

if __name__ == "__main__":
    main()