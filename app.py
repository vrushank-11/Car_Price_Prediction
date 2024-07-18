from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('best_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load your dataset
df = pd.read_csv('dataset/cardekho.csv')


def get_unique_values():
    unique_values = {}
    unique_values['name'] = sorted(df['name'].unique().tolist())
    unique_values['year'] = sorted(df['year'].unique().tolist())
    unique_values['fuel'] = df['fuel'].unique().tolist()
    return unique_values


@app.route('/')
def index():
    unique_values = get_unique_values()
    return render_template('index.html', unique_values=unique_values)


@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    year = int(request.form['year'])
    km_driven = float(request.form['km_driven'])
    mileage = float(request.form['mileage'])
    engine = float(request.form['engine'])
    max_power = float(request.form['max_power'])
    fuel = request.form['fuel']

    # Use existing label encoders if they exist
    if 'name' in label_encoders:
        name_encoder = label_encoders['name']
    else:
        return render_template('index.html', error='Error: Label encoders not found.', unique_values=get_unique_values())

    if 'fuel' in label_encoders:
        fuel_type_encoder = label_encoders['fuel']
    else:
        return render_template('index.html', error='Error: Label encoders not found.', unique_values=get_unique_values())

    # Ensure the input values are in the known categories
    unique_values = get_unique_values()
    if fuel not in fuel_type_encoder.classes_:
        return render_template('index.html', error=f'Error: Fuel Type "{fuel}" not in training data.', unique_values=unique_values)
    if name not in name_encoder.classes_:
        return render_template('index.html', error=f'Error: Car Name "{name}" not in training data.', unique_values=unique_values)

    # Encode categorical variables
    fuel_type_encoded = fuel_type_encoder.transform([fuel])[0]
    name_encoded = name_encoder.transform([name])[0]

    # Assuming your original input features are in the order expected by the model
    input_features = np.array(
        [[name_encoded, year, km_driven, fuel_type_encoded, 0, 0, mileage, engine, max_power, 0, 0]])

    # If you want to include feature names, you can create a DataFrame
    feature_names = ['name', 'year', 'km_driven', 'fuel', 'seller_type',
                     'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    input_features_df = pd.DataFrame(input_features, columns=feature_names)

    # Make prediction
    predicted_price = model.predict(input_features_df)
    # Return prediction
    return render_template('index.html', prediction=f'Predicted Price : ₹{predicted_price[0]:.2f}', unique_values=get_unique_values())


if __name__ == '__main__':
    app.run(debug=True)
