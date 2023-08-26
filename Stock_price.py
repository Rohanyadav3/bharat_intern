import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("your_dataset.csv")

# Preprocess 'Close/Last' column to remove non-numeric characters
data["Close/Last"] = data["Close/Last"].str.replace('$', '').astype(float)

# Define features (X) and target (y)
X = data.drop(columns=["Date", "Close/Last"])  # Exclude 'Date' column
y = data["Close/Last"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Take user input for the day's details
open_price = float(input("Enter the Open price: "))
high = float(input("Enter the High price: "))
low = float(input("Enter the Low price: "))

# Make predictions using the trained model
new_data = pd.DataFrame({
    "Open": [open_price],
    "High": [high],
    "Low": [low]
})
prediction = model.predict(new_data)

# Print the predicted price
print("Predicted Close/Last Price:", prediction[0])
