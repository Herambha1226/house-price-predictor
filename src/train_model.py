import os
import json
import numpy as np 
import pandas as pd  
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error



class LinearRegressionModel:

    def __init__(self,data):
        self.data = data 
        numeric_data = data.select_dtypes(include=["number"]).dropna()
        self.x = numeric_data.drop("SalePrice",axis=1)
        self.y = numeric_data["SalePrice"]
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,test_size=0.2,random_state=42)
        self.model = None

    def model_training(self):
        self.model = LinearRegression()
        self.model.fit(self.x_train,self.y_train)
        print(f"Train Score : {self.model.score(self.x_train,self.y_train):.4f}")
        print(f"Test Score : {self.model.score(self.x_test, self.y_test):.4f}")

        os.makedirs("models",exist_ok=True)
        joblib.dump(self.model,"models/linear_regression.pkl")
        print("Model Saved to models/linear_regression.pkl")
    
    def predict(self,input_data:pd.DataFrame) -> np.ndarray:
        if self.model is None:
            if not os.path.exists("models/linear_regression.pkl"):
                raise FileNotFoundError("Model not trained yet. Run model_training() first.")
            self.model = joblib.load("models/linear_regression.pkl")
        return self.model.predict(input_data)
    
    def evaluate(self):
        y_pred_log = self.predict(self.x_test)

        # log-scale metrics (for Kaggle RMSLE)
        r2   = self.model.score(self.x_test, self.y_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_log))
        mae  = mean_absolute_error(self.y_test, y_pred_log)

        # real dollar metrics (for humans)
        y_pred_actual = np.expm1(y_pred_log)
        y_test_actual = np.expm1(self.y_test)
        rmse_dollars  = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        mae_dollars   = mean_absolute_error(y_test_actual, y_pred_actual)

        print("─" * 35)
        print(f"  Train R²         : {self.model.score(self.x_train, self.y_train):.4f}")
        print(f"  Test  R²         : {r2:.4f}")
        print(f"  RMSE (log scale) : {rmse:.4f}")
        print(f"  MAE  (log scale) : {mae:.4f}")
        print(f"  RMSE (dollars)   : ${rmse_dollars:,.0f}")
        print(f"  MAE  (dollars)   : ${mae_dollars:,.0f}")
        print("─" * 35)

        data = {
            "model" : "LinearRegressionModel",
            "Train R2" : f"{self.model.score(self.x_train, self.y_train):.4f}",
            "Test R2" : f"{r2:.4f}",
            "RMSE (log scale)" : f"{rmse:.4f}",
            "MAE" : f"{mae:.4f}",
            "RMSE (dollars)" : f"${rmse_dollars:,.0f}",
            "MAE (dollars)" : f"${mae_dollars:,.0f}"
        }

        json_str = json.dumps(data,indent=4)
        with open("models_evaluation.json",'a') as f:
            f.write(json_str)

        
    def main(self):
        self.model_training()
        self.evaluate()


class RandomForestModel:
    def __init__(self,data):
        self.data = data 
        numeric_data = data.select_dtypes(include=["number"]).dropna()
        self.x = numeric_data.drop("SalePrice",axis=1)
        self.y = numeric_data["SalePrice"]
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,test_size=0.2,random_state=42)
        self.model = None

    def model_training(self):
        self.model = RandomForestRegressor(n_estimators=100,max_depth=None,random_state=42,n_jobs=-1)
        self.model.fit(self.x_train,self.y_train)
        print(f"Train Score : {self.model.score(self.x_train,self.y_train):.4f}")
        print(f"Test Score : {self.model.score(self.x_test, self.y_test):.4f}")

        os.makedirs("models",exist_ok=True)
        joblib.dump(self.model,"models/random_forest.pkl")
        print("Model Saved to models/random_forest.pkl")
    
    def predict(self,input_data:pd.DataFrame) -> np.ndarray:
        if self.model is None:
            if not os.path.exists("models/random_forest.pkl"):
                raise FileNotFoundError("Model not trained yet. Run model_training() first.")
            self.model = joblib.load("models/random_forest.pkl")
        return self.model.predict(input_data)
    
    def evaluate(self):
        y_pred_log = self.predict(self.x_test)

        # log-scale metrics (for Kaggle RMSLE)
        r2   = self.model.score(self.x_test, self.y_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_log))
        mae  = mean_absolute_error(self.y_test, y_pred_log)

        # real dollar metrics (for humans)
        y_pred_actual = np.expm1(y_pred_log)
        y_test_actual = np.expm1(self.y_test)
        rmse_dollars  = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        mae_dollars   = mean_absolute_error(y_test_actual, y_pred_actual)

        print("─" * 35)
        print(f"  Train R²         : {self.model.score(self.x_train, self.y_train):.4f}")
        print(f"  Test  R²         : {r2:.4f}")
        print(f"  RMSE (log scale) : {rmse:.4f}")
        print(f"  MAE  (log scale) : {mae:.4f}")
        print(f"  RMSE (dollars)   : ${rmse_dollars:,.0f}")
        print(f"  MAE  (dollars)   : ${mae_dollars:,.0f}")
        print("─" * 35)

        data = {
            "model" : "RandomForestModel",
            "Train R2" : f"{self.model.score(self.x_train, self.y_train):.4f}",
            "Test R2" : f"{r2:.4f}",
            "RMSE (log scale)" : f"{rmse:.4f}",
            "MAE" : f"{mae:.4f}",
            "RMSE (dollars)" : f"${rmse_dollars:,.0f}",
            "MAE (dollars)" : f"${mae_dollars:,.0f}"
        }

        json_str = json.dumps(data,indent=4)
        with open("models_evaluation.json",'a') as f:
            f.write(json_str)
        
    def main(self):
        self.model_training()
        self.evaluate()
    
class XGBoostModel:
    def __init__(self,data):
        self.data = data 
        numeric_data = data.select_dtypes(include=["number"]).dropna()
        self.x = numeric_data.drop("SalePrice",axis=1)
        self.y = numeric_data["SalePrice"]
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,test_size=0.2,random_state=42)
        self.model = None

    def model_training(self):
        self.model = XGBRegressor(
            n_estimators=300,
            learning_rate = 0.05,
            max_depth=6,
            subsameple=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(self.x_train,self.y_train)
        print(f"Train Score : {self.model.score(self.x_train,self.y_train):.4f}")
        print(f"Test Score : {self.model.score(self.x_test, self.y_test):.4f}")

        os.makedirs("models",exist_ok=True)
        joblib.dump(self.model,"models/xgboost.pkl")
        print("Model Saved to models/xgboost.pkl")
    
    def predict(self,input_data:pd.DataFrame) -> np.ndarray:
        if self.model is None:
            if not os.path.exists("models/xgboost.pkl"):
                raise FileNotFoundError("Model not trained yet. Run model_training() first.")
            self.model = joblib.load("models/xgboost.pkl")
        return self.model.predict(input_data)
    
    def evaluate(self):
        y_pred_log = self.predict(self.x_test)

        # log-scale metrics (for Kaggle RMSLE)
        r2   = self.model.score(self.x_test, self.y_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_log))
        mae  = mean_absolute_error(self.y_test, y_pred_log)

        # real dollar metrics (for humans)
        y_pred_actual = np.expm1(y_pred_log)
        y_test_actual = np.expm1(self.y_test)
        rmse_dollars  = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        mae_dollars   = mean_absolute_error(y_test_actual, y_pred_actual)

        print("─" * 35)
        print(f"  Train R²         : {self.model.score(self.x_train, self.y_train):.4f}")
        print(f"  Test  R²         : {r2:.4f}")
        print(f"  RMSE (log scale) : {rmse:.4f}")
        print(f"  MAE  (log scale) : {mae:.4f}")
        print(f"  RMSE (dollars)   : ${rmse_dollars:,.0f}")
        print(f"  MAE  (dollars)   : ${mae_dollars:,.0f}")
        print("─" * 35)

        data = {
            "model" : "XGBoostModel",
            "Train R2" : f"{self.model.score(self.x_train, self.y_train):.4f}",
            "Test R2" : f"{r2:.4f}",
            "RMSE (log scale)" : f"{rmse:.4f}",
            "MAE" : f"{mae:.4f}",
            "RMSE (dollars)" : f"${rmse_dollars:,.0f}",
            "MAE (dollars)" : f"${mae_dollars:,.0f}"
        }

        json_str = json.dumps(data,indent=4)
        with open("models_evaluation.json",'a') as f:
            f.write(json_str)
        
    def main(self):
        self.model_training()
        self.evaluate()


if __name__ == "__main__":
    data = pd.read_csv("data/processed/clean_data.csv")

    obj = LinearRegressionModel(data=data)
    obj.main()

    obj1 = RandomForestModel(data=data)
    obj1.main()

    obj2 = XGBoostModel(data=data)
    obj2.main()


        