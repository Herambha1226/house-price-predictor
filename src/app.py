from flask import render_template, Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # use must before the pyplot import
import matplotlib.pyplot as plt 
import io
import base64
import json
from sklearn.metrics import r2_score


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

with open("models_evaluation.json","r") as f:
    evaluation_data = json.load(f)
r2_scores = [float(m["Test R2"]) for m in evaluation_data]

rmse_vals = [int(m["RMSE (dollars)"].replace("$","").replace(",",""))
             for m in evaluation_data]
model_names = [m["model"].replace("Model", "") for m in evaluation_data]
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

        #print(f"INPUT → quality:{quality} area:{area} bed:{bedrooms} bath:{bathrooms} year:{year_pred}")

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
        #print(f"Input shape : {input_df.shape}")   # must be (1, 42)

        pred_xgb    = float(np.expm1(model_xgb.predict(input_df)[0]))
        pred_random = float(np.expm1(model_random.predict(input_df)[0]))
        pred_linear = float(np.expm1(model_linear.predict(input_df)[0]))

        print(f"XGBoost     : ${pred_xgb:,.0f}")
        print(f"RandomForest: ${pred_random:,.0f}")
        print(f"Linear      : ${pred_linear:,.0f}")

        def safe_value(val,cap=1_000_000):
            return min(val,cap)
        
        pred_xgb = safe_value(pred_xgb)
        pred_random = safe_value(pred_random)
        pred_linear = safe_value(pred_linear)

    
        fig,ax = plt.subplots(1,2,figsize=(12,5))

        # prediction 
        models = ["XGBoost","RandomForest","Linear"]
        prediction = [pred_xgb,pred_random,pred_linear]

        sorted_data = sorted(zip(models,prediction),key=lambda x: x[1])
        m1,p1 = zip(*sorted_data)

        ax[0].barh(m1,p1)

        for i,v in enumerate(p1):
            ax[0].text(v + max(p1)*0.01,i,f"${v:,.0f}",va="center")
        ax[0].set_title("Prediction Comparision")
        ax[0].set_xlabel("Price (USD)")

        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)

        #  model accuracy
        ax[1].bar(model_names,r2_scores)

        for i,v in enumerate(r2_scores):
            ax[1].text(i,v + 0.002,f"{v:.3f}",ha="center")
        
        ax[1].set_title("Model Accuracy (Test R2)")
        ax[1].set_ylabel("R2 Score")

        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf,format="png",dpi=128,bbox_inches="tight")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()


        return jsonify({
            "success"    : True,
            "prediction" : f"${pred_xgb:,.0f}",
            "xgboost"    : f"${pred_xgb:,.0f}",
            "random"     : f"${pred_random:,.0f}",
            "linear"     : f"${pred_linear:,.0f}",
            "low"        : f"${pred_xgb * 0.92:,.0f}",
            "high"       : f"${pred_xgb * 1.08:,.0f}",
            "chart"      : img_b64
        })

    except Exception as e:
        print(f"PREDICT ERROR: {e}")
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)