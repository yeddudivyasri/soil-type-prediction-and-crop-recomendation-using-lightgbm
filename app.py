from flask import Flask, request, render_template, jsonify
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and prepare the dataset
df = pd.read_csv("soil_crop_dataset.csv")
le_soil = LabelEncoder()
le_crop = LabelEncoder()
df["Soil_Type_Label"] = le_soil.fit_transform(df["Soil_Type"])
df["Crop"] = le_crop.fit_transform(df["Recommended_Crop"])
features = ["pH", "Nitrogen", "Phosphorus", "Potassium", "Moisture", "Temperature", "Rainfall"]
X = df[features]
y_soil = df["Soil_Type_Label"]
y_crop = df["Crop"]
X_train, X_test, y_soil_train, y_soil_test = train_test_split(X, y_soil, test_size=0.2, random_state=42)
_, _, y_crop_train, y_crop_test = train_test_split(X, y_crop, test_size=0.2, random_state=42)
soil_model = lgb.LGBMClassifier()
crop_model = lgb.LGBMClassifier()
soil_model.fit(X_train, y_soil_train)
crop_model.fit(X_train, y_crop_train)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    soil_pred = soil_model.predict(input_df)[0]
    crop_pred = crop_model.predict(input_df)[0]
    soil_type = le_soil.inverse_transform([soil_pred])[0]
    crop_name = le_crop.inverse_transform([crop_pred])[0]
    return jsonify({"soil_type": soil_type, "recommended_crop": crop_name})

if __name__ == "__main__":
    app.run(debug=True)
