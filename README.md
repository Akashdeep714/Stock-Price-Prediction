# 📈 Stock Price Prediction using LSTM

This project demonstrates how to build and train a Long Short-Term Memory (LSTM) model to predict stock prices using historical time series data. It uses deep learning techniques to learn temporal dependencies and forecast future stock prices with reasonable accuracy.

---

## 🚀 Features

- Scales time series data using `MinMaxScaler`
- Splits dataset into training and testing segments
- Creates sequences of historical prices for LSTM input
- Builds a deep LSTM model using `TensorFlow` and `Keras`
- Trains and evaluates the model with metrics like RMSE
- Visualizes actual vs predicted prices for both training and testing datasets

---

## 📂 Project Structure

```

📁 stock\_price\_prediction
│
├── stock\_price\_prediction.ipynb   # Jupyter Notebook with complete code
├── README.md                      # Project documentation (this file)
└── requirements.txt               # (Optional) List of dependencies

````

---

## 🧠 Technologies Used

- Python 3.x
- TensorFlow / Keras
- Numpy, Pandas
- Scikit-learn
- Matplotlib
- (Optional) yfinance or pandas_datareader for live data

---

## 📉 How It Works

### 1. Data Preprocessing
- Scales data to range (0, 1) using `MinMaxScaler`
- Creates time-stepped sequences for LSTM input
- Splits into training and test sets (typically 70% training)

### 2. Model Architecture
- Stacked LSTM layers with dropout (if added)
- Dense output layer for predicting closing price

### 3. Training
- Trained for 100 epochs with batch size 64
- Uses MSE loss and Adam optimizer

### 4. Evaluation
- Root Mean Squared Error (RMSE)
- Visual plots comparing predicted vs actual


## ⚙️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/stock_price_prediction.git
   cd stock_price_prediction
````

2. **Install dependencies**
   *(You can use virtualenv or conda if preferred)*

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not available, install manually:

   ```bash
   pip install numpy pandas matplotlib scikit-learn tensorflow
   ```

3. **Run the notebook**

   ```bash
   jupyter notebook stock_price_prediction.ipynb
   ```

---

## 📌 Notes

* Make sure you have a GPU enabled (optional) to speed up training.
* The model currently uses only one feature (e.g., closing price). You can extend it to include volume, open, high, low, etc.
* Consider experimenting with different time steps, batch sizes, and model depths.

---

## 🧪 Future Improvements

* Add live stock data fetching using `yfinance`
* Save and load model using `.h5` format
* Extend to multivariate prediction
* Deploy as a web app using Streamlit

---

## 🤝 Contribution

Feel free to fork the repo, raise issues, and submit pull requests. Contributions are welcome!

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

```
