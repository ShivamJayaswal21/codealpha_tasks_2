import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Dataset
df = pd.read_csv("car data.csv")

# Remove extra spaces in column names
df.columns = df.columns.str.strip()

print("Columns in dataset:")
print(df.columns)

# 2. Feature Engineering (Only if Year exists)
if 'Year' in df.columns:
    df['Car_Age'] = 2026 - df['Year']
    df.drop(['Year'], axis=1, inplace=True)

# 3. Encode All Categorical Columns Automatically
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# 4. Define Target Column (Change if different)
if 'Selling_Price' in df.columns:
    target = 'Selling_Price'
elif 'Price' in df.columns:
    target = 'Price'
else:
    print("⚠ Please check your price column name")
    exit()

X = df.drop(target, axis=1)
y = df[target]

# 5. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Prediction
y_pred = model.predict(X_test)

# 8. Evaluation
print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# 9. Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()
