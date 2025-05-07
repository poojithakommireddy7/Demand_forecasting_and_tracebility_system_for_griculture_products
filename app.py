from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

app = Flask(__name__)

# Preprocessing and Model Training (as before)
file_path = Path(r'C:/Users/M TEJASWI/OneDrive/Desktop/Project/Python/Agriculture_dataset.xlsx')
if not file_path.exists():
    raise FileNotFoundError(f"The file at {file_path} does not exist.")

# Load and preprocess the data
data = pd.ExcelFile(file_path)
df = data.parse('Sheet1')

df['Season'] = df['Season'].str.lower()
df['Region'] = df['Region'].str.lower()
df['Crop'] = df['Crop'].str.lower()

season_encoder = LabelEncoder()
region_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
crop_prediction_encoder = LabelEncoder()

df['Season'] = season_encoder.fit_transform(df['Season'])
df['Region'] = region_encoder.fit_transform(df['Region'])
df['Crop'] = crop_encoder.fit_transform(df['Crop'])
df['Crop Prediction'] = crop_prediction_encoder.fit_transform(df['Crop Prediction'])

crop_mapping = dict(zip(range(len(crop_prediction_encoder.classes_)), crop_prediction_encoder.classes_))

X = df[['Season', 'Region', 'Crop']]
y = df['Crop Prediction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

gb_accuracy = accuracy_score(y_test, gb_model.predict(X_test))
dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))

def forecast_crop_prediction(season, region, crop):
    season = season.lower()
    region = region.lower()
    crop = crop.lower()

    if season not in season_encoder.inverse_transform(range(len(season_encoder.classes_))):
        raise ValueError(f"Season '{season}' not found in training data.")
    if region not in region_encoder.inverse_transform(range(len(region_encoder.classes_))):
        raise ValueError(f"Region '{region}' not found in training data.")
    if crop not in crop_encoder.inverse_transform(range(len(crop_encoder.classes_))):
        raise ValueError(f"Crop '{crop}' not found in training data.")

    encoded_season = season_encoder.transform([season])[0]
    encoded_region = region_encoder.transform([region])[0]
    encoded_crop = crop_encoder.transform([crop])[0]

    gb_prediction = gb_model.predict([[encoded_season, encoded_region, encoded_crop]])[0]
    dt_prediction = dt_model.predict([[encoded_season, encoded_region, encoded_crop]])[0]

    final_prediction = gb_prediction if gb_accuracy > dt_accuracy else dt_prediction
    return crop_mapping[final_prediction].capitalize()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        season = request.form['season']
        region = request.form['region']
        crop = request.form['crop']
        prediction = forecast_crop_prediction(season, region, crop)
        return render_template('index.html', prediction=prediction)
    except ValueError as e:
        return render_template('index.html', error=str(e))
    except Exception as e:
        return render_template('index.html', error='An error occurred.')

if __name__ == '__main__':
    app.run(debug=True)
