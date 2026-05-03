import pandas as pd 
import matplotlib.pyplot as plt

class ModelComparision:
    def __init__(self,data):
        self.data = data
    
    # Covert numeric columns
    def cvt_numeric(self):
        self.data["Train R2"] = self.data["Train R2"].astype(float)
        self.data["Test R2"] = self.data["Test R2"].astype(float)
        self.data["RMSE (log scale)"] = self.data["RMSE (log scale)"].astype(float)
        self.data["MAE"] = self.data["MAE"].astype(float)

        self.data["RMSE (dollars)"] = self.data["RMSE (dollars)"].replace('[\\$,]','',regex=True).astype(float)
        self.data["MAE (dollars)"] = self.data["MAE (dollars)"].replace('[\\$,]','',regex=True).astype(float)

    def comparision_graphs(self):
        fig,ax = plt.subplots(2,2,figsize=(12,8))

        self.data.set_index("model")[["Train R2","Test R2"]].plot(kind="bar",ax=ax[0,0])
        ax[0,0].set_title("R2 Score Comparision")
        ax[0,0].set_ylabel("R2 Score")
        ax[0,0].tick_params(rotation=0)
       

        self.data.set_index("model")[["RMSE (log scale)","MAE"]].plot(kind="bar",ax=ax[0,1])
        ax[0,1].set_title("Error Comparision (Log Scale)")
        ax[0,1].set_ylabel("Error")
        ax[0,1].tick_params(rotation=0)
        

        self.data.set_index("model")[["RMSE (dollars)","MAE (dollars)"]].plot(kind="bar",ax=ax[1,0])
        ax[1,0].set_title("Error in Dollars (Real Impact)")
        ax[1,0].set_ylabel("USD")
        ax[1,0].tick_params(rotation=0)
        

        self.data["Overfitting Gap"] = self.data["Train R2"] - self.data["Test R2"]
        self.data.set_index("model")["Overfitting Gap"].plot(kind="bar",color="orange",ax=ax[1,1])
        ax[1,1].set_title("Overfitting Gap (Train - Test)")
        ax[1,1].set_ylabel("Difference")
        ax[1,1].tick_params(rotation=0)

        plt.tight_layout()
        plt.savefig("model_perform_visual/Models_Performance.png")
        plt.show()
        
    
    def main(self):
        self.cvt_numeric()
        self.comparision_graphs()


if __name__ == "__main__":
    data = pd.read_json("models_evaluation.json")
    obj = ModelComparision(data=data)
    obj.main()

        