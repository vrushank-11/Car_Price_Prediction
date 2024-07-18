# Car Price Prediction Web Application

This web application predicts the selling price of cars based on user input features. The application is built using Flask, and the machine learning model is trained using the Random Forest algorithm.

## Supervised by
[Prof. Agughasi Victor Ikechukwu](https://github.com/Victor-Ikechukwu), (Assistant Professor) Department of CSE, MIT Mysore

## Collaborators
- 4MH21CS118 [Vrushank Gowda K](https://github.com/vrushank-11)
- 4MH21CS101 [Subhash H T](https://github.com/Subhashdarya)
- 4MH21CS098 [Skanda N](https://github.com/Skanda2809)

## Features

- User input for car details including name, year, kilometers driven, mileage, engine, max power, and fuel type.
- Prediction of car selling price based on the input features.
- User-friendly interface with form validation.

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/car-price-prediction.git
    cd car-price-prediction
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate    # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Flask application:**
    ```sh
    flask run


```
dataset
│   cardekho.csv
│
images
│   carprice.png
│   pair_plot.png
│
static
│   styles.css
│
templates
│   index.html
│
app.py
best_model.pkl
carprice-prediction.ipynb
carprice-prediction.py
label_encoders.pkl
README.md
requirements.txt
```

## File Structure

- `app.py`: Main application file that contains the Flask server code.
- `templates/index.html`: HTML template for the web interface.
- `static/css/style.css`: Optional CSS file for styling the web interface.
- `dataset/cardekho.csv`: Dataset file used for training the model.
- `best_model.pkl`: Trained machine learning model.
- `label_encoders.pkl`: Label encoders for categorical features.
- `requirements.txt`: List of Python packages required to run the application.

## Usage

1. **Access the web application:**
    Open your web browser and go to `http://127.0.0.1:5000/`.

2. **Enter car details:**
    Fill out the form with the car details including name, year, kilometers driven, mileage, engine, max power, and fuel type.

3. **Get prediction:**
    Click on the 'Predict' button to get the predicted selling price of the car.

## Example Form Fields

- **Car Name:** Select the car name from the dropdown list.
- **Year:** Enter the manufacturing year of the car.
- **Kilometers Driven:** Enter the total kilometers driven.
- **Mileage:** Enter the mileage of the car.
- **Engine:** Enter the engine capacity.
- **Max Power:** Enter the maximum power of the car.
- **Fuel Type:** Select the fuel type from the dropdown list.

## Model Training

The machine learning model is trained using the Random Forest algorithm. The dataset (`cardekho.csv`) includes features like name, year, kilometers driven, mileage, engine, max power, and fuel type. The model and label encoders are saved as `best_model.pkl` and `label_encoders.pkl` respectively.

### Notebook for Model Training

The Jupyter notebook used for data preprocessing, model training, and evaluation is included. Follow the steps in the notebook to understand the data processing and model training workflow.

## Requirements

- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Seaborn
- Matplotlib

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

---

### Example `requirements.txt`

Here is an example `requirements.txt` for your project:

```
Flask==2.0.2
pandas==1.3.3
numpy==1.21.2
scikit-learn==0.24.2
joblib==1.0.1
seaborn==0.11.2
matplotlib==3.4.3
```

### Example `index.html`

Here is an example of the `index.html` file to use with the updated Flask app:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Car Price Prediction</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <h1>Car Price Prediction</h1>
        <form action="{{ url_for('predict') }}" method="post">
            <label for="name">Car Name:</label>
            <select id="name" name="name">
                {% for value in unique_values['name'] %}
                <option value="{{ value }}">{{ value }}</option>
                {% endfor %}
            </select>
            <br>
            <label for="year">Year:</label>
            <select id="year" name="year">
                {% for value in unique_values['year'] %}
                <option value="{{ value }}">{{ value }}</option>
                {% endfor %}
            </select>
            <br>
            <label for="km_driven">Kilometers Driven:</label>
            <input type="number" id="km_driven" name="km_driven" step="0.01" required>
            <br>
            <label for="mileage">Mileage:</label>
            <input type="number" id="mileage" name="mileage" step="0.01" required>
            <br>
            <label for="engine">Engine:</label>
            <input type="number" id="engine" name="engine" step="0.01" required>
            <br>
            <label for="max_power">Max Power:</label>
            <input type="number" id="max_power" name="max_power" step="0.01" required>
            <br>
            <label for="fuel">Fuel Type:</label>
            <select id="fuel" name="fuel">
                {% for value in unique_values['fuel'] %}
                <option value="{{ value }}">{{ value }}</option>
                {% endfor %}
            </select>
            <br>
            <button type="submit">Predict</button>
        </form>
        {% if prediction %}
        <div class="result">
            <h2>{{ prediction }}</h2>
        </div>
        {% endif %}
        {% if error %}
        <div class="error">
            <h2>{{ error }}</h2>
        </div>
        {% endif %}
    </div>
</body>
</html>
```

### Input form and Prediction Result
![ Input For and Prediction Result](https://github.com/vrushank-11/Car_Price_Prediction/blob/main/images/carprice.png)

### Diabetes Visualization

#### EDA - Pairplot
![Pairplot](https://github.com/vrushank-11/Car_Price_Prediction/blob/main/images/pair_plot.png)

## Conclusion

This project demonstrates the application of data science techniques to predict car prices using machine learning. By performing thorough exploratory data analysis (EDA), training multiple models, and deploying the best model using Flask, we provide a practical tool for diabetes prediction.

## License
This project is licensed under the MIT License.
