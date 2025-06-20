Soil Type Prediction & Crop Recommendation using LightGBM
---------------------------------------------------------
Soil Type Prediction & Crop Recommendation is a web-based machine learning tool built with Python (Flask) and LightGBM. It helps predict the soil type and suggests the most suitable crop based on key agricultural inputs like soil nutrients, moisture, temperature, and rainfall.

This project simplifies precision farming by offering farmers or agriculturalists a quick and intuitive way to optimize crop selection for their land conditions.

Features
--------
 Easy form to input soil and climate parameters (pH, N, P, K, moisture, temperature, rainfall)

 Predicts soil type using a trained LightGBM model

 Recommends the best crop for the predicted soil type

 Real-time results with clean web interface (HTML/CSS + Flask backend)

 Fast, lightweight, and efficient model performance

📦 Installation
Clone the repository

git clone https://github.com/yeddudivyasri/soil-type-prediction-and-crop-recomendation-using-lightgbm.git
Navigate to the project directory

cd SoilCropPrediction
Install the required dependencies

pip install -r requirements.txt
 Run the Application
 ------------------

python app.py
Once the server starts, open your browser and go to:
http://127.0.0.1:5000/

Enter the input values → Click Predict → Get instant soil type and crop recommendation.

Tech Stack
----------
Python

Flask

LightGBM

Pandas

HTML5/CSS3/JavaScript

License
This project is licensed under the MIT License. See the LICENSE file for details.

