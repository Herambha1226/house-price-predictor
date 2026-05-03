import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

class Preprocessing:
    def __init__(self):
        self.data = pd.read_csv("data/raw/train.csv")

    def handle_missing_values(self):
        data = self.data.copy()

        num_cols = data.select_dtypes(include=["number"]).columns
        cat_cols = data.select_dtypes(include=["object", "string"]).columns

        # Numeric → mean
        for col in num_cols:
            data[col] = data[col].fillna(data[col].mean())

        # Categorical → mode
        for col in cat_cols:
            data[col] = data[col].fillna(data[col].mode()[0])

        self.data = data

    def remove_outlier(self):
        data = self.data.copy()

        Q1 = data['SalePrice'].quantile(0.25)
        Q3 = data['SalePrice'].quantile(0.75)
        IQR = Q3 - Q1

        condition = ~(
            (data['SalePrice'] < (Q1 - 1.5 * IQR)) |
            (data['SalePrice'] > (Q3 + 1.5 * IQR))
        )

        self.data = data.loc[condition].reset_index(drop=True)

    def feature_engineering(self):
        data = self.data.copy()

        data["TotalSF"] = data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"]
        data["HouseAge"] = data["YrSold"] - data["YearBuilt"]
        data["RemodAge"] = data["YrSold"] - data["YearRemodAdd"]
        data["TotalBath"] = (
            data["FullBath"] + 0.5 * data["HalfBath"] +
            data["BsmtFullBath"] + 0.5 * data["BsmtHalfBath"]
        )

        data["HasGarage"] = (data["GarageArea"] > 0).astype(int)
        data["HasPool"] = (data["PoolArea"] > 0).astype(int)
        data["HasFireplace"] = (data["Fireplaces"] > 0).astype(int)

        data["OverallScore"] = data["OverallQual"] * data["OverallCond"]
        data["PricePerSF"] = data["SalePrice"] / (data["TotalSF"] + 1)  # avoid divide by zero

        self.data = data

    def encode_categorical(self):
        data = self.data.copy()
        data = pd.get_dummies(data, drop_first=True)
        self.data = data

    def log_transform_target(self):
        data = self.data.copy()
        data['SalePrice'] = np.log1p(data['SalePrice'])
        self.data = data

    def remove_low_variance(self, threshold=0.01):
        data = self.data.copy()

        num_cols = data.select_dtypes(include=["number"]).columns

        selector = VarianceThreshold(threshold)
        num_data = selector.fit_transform(data[num_cols])

        selected_cols = num_cols[selector.get_support()]

        self.data = pd.DataFrame(num_data, columns=selected_cols, index=data.index)

    def remove_high_correlation(self, threshold=0.9):
        data = self.data.copy()

        corr_matrix = data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]

        self.data = data.drop(columns=drop_cols)

    def scale_features(self):
        data = self.data.copy()

        scaler = StandardScaler()

        if 'SalePrice' in data.columns:
            X = data.drop(columns=['SalePrice'])
            y = data['SalePrice']

            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=data.index)

            self.data = pd.concat([X_scaled, y], axis=1)
        else:
            self.data = pd.DataFrame(
                scaler.fit_transform(data),
                columns=data.columns,
                index=data.index
            )

    def save_data(self, path="data/processed/clean_data.csv"):
        self.data.to_csv(path, index=False)
        print(f"✅ File saved to {path}")

    def main(self):
        self.handle_missing_values()
        self.remove_outlier()
        self.feature_engineering()
        self.encode_categorical()
        self.log_transform_target()
        self.remove_low_variance()
        self.remove_high_correlation()
        self.scale_features()
        self.save_data()


if __name__ == "__main__":
    obj = Preprocessing()
    obj.main()
