# Heart Failure Prediction Web App

This project is a **Heart Failure Prediction** web application built using **Flask** and **Python**. The goal of this web app is to predict the likelihood of heart failure based on various health-related factors using a trained machine learning model. Note that this model does not accurately predict the results, the main goal is to make web app using flask framework.

## Example

**Input:**

![Screenshot of main page](https://github.com/dhavaldalvi/heart-failure-prediction-web-application/blob/main/screenshots/Index.JPG?raw=true)

**Prediction:**

![Screenshot of result page](https://github.com/dhavaldalvi/heart-failure-prediction-web-application/blob/main/screenshots/Result.JPG?raw=true)
---

## Features

- **Predict Heart Failure Risk**: Enter health parameters such as age, cholesterol, maximum heart rate, etc., and the model will predict the likelihood of heart failure.
- **Real-time Prediction**: The model is applied in real-time through a simple and interactive web interface.
- **Interactive Web Interface**: Developed with Flask to allow users to input data via a clean and user-friendly form.
- **Model Integration**: Machine learning model (trained on heart failure datasets) is integrated into the backend for prediction.

---

## Technologies Used

- **Python** (for the backend and model)
- **Flask** (for creating the web application)
- **scikit-learn** (for machine learning model)
- **Pandas** and **NumPy** (for data manipulation)
- **HTML/CSS** (for the front-end)
- **Pickle** (to load and save the trained machine learning model)

---

## Installation

### Prerequisites

Before running the application, make sure you have the following installed:

- **Python 3.12** 
- **pip** (Python's package installer)

### Or

- **Anaconda** or **Miniconda** (for managing Python environments)

### 1. Create the Environment

Using `venv` of **Python**. Run the following command in **Command Prompt** or **Terminal**.

```bash
python -m venv myenv
```
### Or

Using `conda` of **Anaconda**. Run following command in **Anaconda Prompt**.

```bash
conda create --name myenv
```
`myenv` is the name of the directory where the virtual environment will be created. You can replace `myenv` with any name you prefer.

### 2. Activating Environment

If using **Python**

In **Windows**
```
.\myenv\Scripts\activate
```
In **MacOS/Linux**
```
source myenv/bin/activate
```
Or if using **Anaconda**

In **Windows**
```
conda activate myenv
```
In **MacOS/Linux**
```
source activate myenv
```

### 3. Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/dhavaldalvi/heart-failure-prediction-web-application.git
```

### 4. Install Dependencies

Navigate to the project directory and install the required packages using pip if using **Python**

```bash
cd heart-failure-prediction-web-app
pip install -r requirements.txt
```
or using conda if using **Anaconda** 

```
cd heart-failure-prediction-web-app
conda install --yes --file requirements.txt
```

The `requirements.txt` file contain the necessary libraries.

### 5. Run the Flask App

To start the application, run the following command:

```bash
python app.py
```

This will start the Flask server and generate some pickle files with extension '*.pkl'. By default, the app will be hosted at `http://127.0.0.1:5000/`.

---

## Usage

Once the app is running, open your browser and navigate to `http://127.0.0.1:5000/`. You will see a form where you can input the following parameters related to heart health:

- **Age**
- **Sex**
- **Chest Pain Type**
- **Resting Blood Pressure**
- **Cholesterol**
- **Fasting Blood Sugar**
- **Resting ECG**
- **Maximum Heart Rate**
- **Exercise Induced Angina**
- **Oldpeak**
- **ST Segment Slope**

After entering the values, click on the "Submit" button, and the model will predict whether the person has a high or low risk of heart failure.

The result will be displayed immediately, and it will indicate the predicted probability of heart failure.

---

## Model Details

- **Dataset**: The model was trained using a publicly available heart failure dataset that includes several health-related features (https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Machine Learning Algorithm**: A machine learning algorithm Random Forest was used to train the model.
- **Model File**: The trained model is saved as a `.pkl` file.




