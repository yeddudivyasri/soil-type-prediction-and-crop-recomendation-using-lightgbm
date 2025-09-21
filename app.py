from flask import Flask, request, jsonify, send_file
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

df = pd.read_csv("soil_crop_dataset_500.csv")

le_soil = LabelEncoder()
le_crop = LabelEncoder()
df["Soil_Type_Label"] = le_soil.fit_transform(df["Soil_Type"])
df["Crop"] = le_crop.fit_transform(df["Recommended_Crop"])

features = ["pH", "Nitrogen", "Phosphorus", "Potassium", "Moisture", "Temperature", "Rainfall"]
X = df[features]
y_soil = df["Soil_Type_Label"]
y_crop = df["Crop"]

# Split dataset for training
X_train, X_test, y_soil_train, y_soil_test, y_crop_train, y_crop_test = train_test_split(
    X, y_soil, y_crop, test_size=0.2, random_state=42
)

# Train LightGBM models
soil_model = lgb.LGBMClassifier()
crop_model = lgb.LGBMClassifier()
soil_model.fit(X_train, y_soil_train)
crop_model.fit(X_train, y_crop_train)

# Routes

# Serve HTML file directly
@app.route("/")
def home():
    return send_file("index.html")

@app.route("/p1.jpg")
def serve_bg():
    return send_file("p1.jpg")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    input_df = pd.DataFrame([data])
    soil_pred = soil_model.predict(input_df)[0]
    crop_pred = crop_model.predict(input_df)[0]

    soil_type = le_soil.inverse_transform([soil_pred])[0]
    crop_name = le_crop.inverse_transform([crop_pred])[0]

    return jsonify({"soil_type": soil_type, "recommended_crop": crop_name})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
