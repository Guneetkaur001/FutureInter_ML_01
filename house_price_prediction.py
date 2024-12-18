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
       
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
    
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['PRICE'] = housing.target
        return df
    
    def preprocess_data(self, df):
       
        df = df.dropna()
        
        X = df.drop('PRICE', axis=1)
        y = df['PRICE']
        
     
        X_scaled = self.scaler.fit_transform(X)
  
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):

        if self.model_type == 'linear':
            self.model = LinearRegression()
        else:
            self.model = DecisionTreeRegressor(random_state=42)
            
        self.model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
    
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            'mean_squared_error': mse,
            'root_mean_squared_error': np.sqrt(mse),
            'r2_score': r2
        }
    
    def predict(self, features):

        if self.model is None:
            raise ValueError("Model hasn't been trained yet!")

        scaled_features = self.scaler.transform(features)
        return self.model.predict(scaled_features)

def main():

    predictor = HousePricePredictor(model_type='linear')
    
 
    print("Loading and preprocessing data...")
    df = predictor.load_data()
    X_train, X_test, y_train, y_test = predictor.preprocess_data(df)
    

    print("Training model...")
    predictor.train_model(X_train, y_train)
    

    print("Evaluating model...")
    metrics = predictor.evaluate_model(X_test, y_test)

    print("\nModel Performance Metrics:")
    print(f"Mean Squared Error: {metrics['mean_squared_error']:.2f}")
    print(f"Root Mean Squared Error: {metrics['root_mean_squared_error']:.2f}")
    print(f"R² Score: {metrics['r2_score']:.2f}")
    
   
    print("\nExample Prediction:")
   
    sample_house = pd.DataFrame([df.drop('PRICE', axis=1).mean()])
    predicted_price = predictor.predict(sample_house)
    print(f"Predicted house price: ${predicted_price[0]:.2f}")

if __name__ == "__main__":
    main()