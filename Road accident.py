import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
data = pd.read_csv("/content/road acc.csv")
X = data[['Time', 'Day_of_week', 'Weather_conditions', 'Type_of_collision', ...]]  # Independent variables
y = data['Accident_severity']  # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, 'linear_regression_model.pkl')
# Step 8: Predict Accident Severity
# Provide an example of using the saved model to predict accident severity for a hypothetical set of independent variables
# Load the model
import joblib
import pandas as pd

loaded_model = joblib.load('linear_regression_model.pkl')
# Prepare hypothetical independent variables
hypothetical_data = pd.DataFrame({'Time': ['08:00'], 'Day_of_week': ['Monday'], 'Weather_conditions': ['Dry'], 'Type_of_collision': ['Rear-end'],})
# Predict accident severity
predicted_severity = loaded_model.predict(hypothetical_data)[0]

print("Predicted accident severity:", predicted_severity)