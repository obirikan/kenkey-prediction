# Sales Prediction using Linear Regression

## Overview
This project uses **Linear Regression** from `scikit-learn` to predict weekly and monthly sales based on past sales data. The model is trained on a dataset containing sales transactions and utilizes **Pickle** for saving the best-trained model.

## Features
- **Preprocessing**: Converts non-numeric data to numeric using `LabelEncoder`.
- **Model Training**: Uses `LinearRegression` to minimize errors.
- **Model Selection**: Runs multiple iterations and saves the best-performing model.
- **Prediction**: Uses the saved model to predict future sales.
- **Visualization**: Plots relationships between features and sales.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Pickle
- NumPy

## Dataset
The dataset used is `kenkey.csv`, which includes:
- `Day` (Day of the transaction)
- `Month` (Month of the transaction)
- `Total_Amt` (Total amount of sales)
- `Qty_Sales` (Quantity sold)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/sales-prediction.git
   cd sales-prediction
   ```
2. Install dependencies:
   ```sh
   pip install pandas scikit-learn matplotlib numpy
   ```
3. Ensure you have the `kenkey.csv` dataset in the same directory.

## Running the Project
Run the script using:
```sh
python sales_prediction.py
```

## How It Works
1. **Load Data**: Reads `kenkey.csv` into a Pandas DataFrame.
2. **Preprocess Data**: Converts categorical data into numerical format.
3. **Split Data**: Divides dataset into training and testing sets.
4. **Train Model**: Uses `LinearRegression` to fit the data and optimize accuracy.
5. **Save Best Model**: Stores the model with the highest accuracy using Pickle.
6. **Make Predictions**: Loads the saved model and predicts future sales.
7. **Visualize Data**: Plots different features against sales for insights.

## Example Output
```
predicted: 150
therefore our weekly sales will be 17822 Ghana Cedis and our monthly sales will be 76380 Ghana Cedis
```

## Visualizations
- **Day vs Sales**
- **Month vs Sales**
- **Total Amount vs Sales**

## Contributions
Feel free to fork and submit a pull request for improvements!

## License
MIT License
