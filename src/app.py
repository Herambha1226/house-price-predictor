from flask import render_template, Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = "Herambha_House_Predictor"

raw_data      = pd.read_csv("data/raw/train.csv")
clean_data    = pd.read_csv("data/processed/clean_data.csv")

neighbourhood = sorted(raw_data["Neighborhood"].dropna().unique().tolist())
garages       = sorted(raw_data["GarageType"].dropna().unique().tolist())
year_built    = sorted([
    int(y) for y in raw_data["YearBuilt"].dropna().unique()
    if y > 1800
], reverse=True)

numeric_data     = clean_data.select_dtypes(include=["number"]).dropna()
pred_columns     = [col for col in numeric_data.columns
                    if col not in ["SalePrice"]]   # only exclude target

avg_price_per_sf = float(clean_data["PricePerSF"].mean())
avg_total_sf     = float(clean_data["TotalSF"].mean())
avg_total_bath   = float(clean_data["TotalBath"].mean())

model_xgb    = joblib.load("models/xgboost.pkl")
model_random = joblib.load("models/random_forest.pkl")
model_linear = joblib.load("models/linear_regression.pkl")

print(f"Feature columns : {len(pred_columns)}")
print(f"Columns         : {pred_columns}")


@app.route('/')
def main():
    return render_template("index.html",
        neighbourhood = neighbourhood,
        year_built    = year_built,
        garages       = garages
    )


@app.route('/predict', methods=["POST"])
def predict():
    try:
        data = request.json

        if data is None:
            return jsonify({"success": False, "error": "No JSON received"})

        quality   = int(data.get('quality', 5))
        area      = float(data.get('area', 0))
        bedrooms  = int(data.get('bedrooms', 0))
        bathrooms = int(data.get('bathrooms', 0))
        year_pred = int(data.get('year_built', 2000))

        print(f"INPUT → quality:{quality} area:{area} bed:{bedrooms} bath:{bathrooms} year:{year_pred}")

        input_dict = {col: 0 for col in pred_columns}

        input_dict['OverallQual']   = quality
        input_dict['GrLivArea']     = area
        input_dict['BedroomAbvGr']  = bedrooms
        input_dict['FullBath']      = bathrooms
        input_dict['YearBuilt']     = year_pred

        input_dict['TotalSF']       = area

        input_dict['TotalBath']     = bathrooms

        input_dict['HasGarage']     = 0

        input_dict['OverallScore']  = quality * 5

        input_dict['PricePerSF']    = avg_price_per_sf

        input_df = pd.DataFrame([input_dict])[pred_columns]
        print(f"Input shape : {input_df.shape}")   # must be (1, 42)

        pred_xgb    = float(np.expm1(model_xgb.predict(input_df)[0]))
        pred_random = float(np.expm1(model_random.predict(input_df)[0]))
        pred_linear = float(np.expm1(model_linear.predict(input_df)[0]))

        print(f"XGBoost     : ${pred_xgb:,.0f}")
        print(f"RandomForest: ${pred_random:,.0f}")
        print(f"Linear      : ${pred_linear:,.0f}")

        return jsonify({
            "success"    : True,
            "prediction" : f"${pred_xgb:,.0f}",
            "xgboost"    : f"${pred_xgb:,.0f}",
            "random"     : f"${pred_random:,.0f}",
            "linear"     : f"${pred_linear:,.0f}",
            "low"        : f"${pred_xgb * 0.92:,.0f}",
            "high"       : f"${pred_xgb * 1.08:,.0f}"
        })

    except Exception as e:
        print(f"PREDICT ERROR: {e}")
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)