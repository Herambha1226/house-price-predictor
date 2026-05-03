import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 

# EDA (Exploratory Data Analysis) of House price predictor
class HousePriceEDA:
    def __init__(self):
        self.data = pd.read_csv("data/raw/train.csv")
    
    def dataset_info(self):
        data = self.data
        print("#"*50)
        print("\t\tDATASET INFORMATION")
        print("#"*50)
        print(f"DataSet Shape / Size : {data.shape}")
        print(f"DataSet Datatypes : {data.dtypes}",end="")
        print("DataSet Info : \n")
        data.info()
    
    def missing_values(self):
        data = self.data 
        missing_val = data.isnull()
        sns.heatmap(missing_val,cbar=True,yticklabels=False,cmap="viridis")
        plt.title("Missing Values Representation With HeatMap")
        plt.show()
    
    def target_distribution(self):
        data = self.data
        fig,ax = plt.subplots(1,2,figsize=(12,4))

        ax[0].hist(data["SalePrice"],bins=50,color="steelblue",edgecolor="white")
        ax[0].set_title("Sales Price - Original (skewed)")
        ax[0].set_xlabel("SalesPrice")

        ax[1].hist(np.log1p(data["SalePrice"]),bins=50,color="teal",edgecolor="white")
        ax[1].set_title("SalesPrice - log transformed (normal)")
        ax[1].set_xlabel("log( SalesPrice )")
        
        plt.tight_layout()
        plt.show()

    
    def comparision_data(self):
        data = self.data
        x = data["GrLivArea"]
        y = data["SalePrice"]
        x1 = data["YearBuilt"]
        y1 = data["SalePrice"]
        
        # GrLivArea vs SalePrice
        plt.scatter(x,y)
        plt.title("GrLivArea vs SalePrice")
        plt.xlabel("GrLivArea")
        plt.ylabel("SalePrice")
        plt.show()

        # YearBuilt vs SalePrice
        plt.scatter(x1,y1)
        plt.title("YearBuilt vs SalePrice")
        plt.xlabel("YearBuilt")
        plt.ylabel("SalePrice")
        plt.show()

        # neighborhood vs price
        sns.boxplot(x = data["Neighborhood"],y = data["SalePrice"])
        plt.xticks(rotation=90)
        plt.xlabel("Neighborhood")
        plt.ylabel("Price")
        plt.title("Neighborhood vs Price")
        plt.show()

        # house style vs price
        sns.boxplot(x=data["HouseStyle"],y=data["SalePrice"])
        plt.xticks(rotation=90)
        plt.xlabel("House Style")
        plt.ylabel("Price")
        plt.title("House Style vs Price")
        plt.show()
    
    def outlier_detection(self):
        data = self.data
        outlier = data[(data["GrLivArea"] > 4000) & (data["SalePrice"] < 300000)]
        print(f"Length Of the Outliers {len(outlier)}")

        plt.scatter(data["GrLivArea"],data["SalePrice"],alpha=0.5,color="steelblue")
        plt.scatter(outlier["GrLivArea"],outlier["SalePrice"],color="red",label = "outlier")
        plt.legend()
        plt.title("Oulier Detection GrlivArea vs SalePrice")
        plt.show()

    def correlation_heatmap(self):
        data = self.data
        numeric = data.select_dtypes(include=[np.number])
        top_features = numeric.corr()["SalePrice"].abs().sort_values(ascending=False)[:15].index

        plt.figure(figsize=(10,8))
        sns.heatmap(numeric[top_features].corr(),annot=True,fmt=".2f",cmap="coolwarm")
        plt.title("Top 15 Features Correlation with SalePrice")
        plt.tight_layout()
        plt.show()
    
    def main(self):
        print(f"Loading House Price Data : \n{self.data}")
        self.dataset_info()
        self.missing_values()
        self.target_distribution()
        self.comparision_data()
        self.outlier_detection()
        self.correlation_heatmap()



if __name__ == "__main__":
    obj = HousePriceEDA()
    obj.dataset_info()

