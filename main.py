import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


car_dataset = pd.read_csv('car data.csv')
car_dataset.head()
print(car_dataset.shape)
print(car_dataset.Fuel_Type.value_counts())


#  Encoding the categorical data

car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)  # Encoding the Fuel type data
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)  # Encoding the seller type data
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)  # Encoding the Transmission data

# Splitting the data and Target
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

# Splitting test data and Training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)

# Model training

    # loading the linear regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

    # Model Evaluation 1
training_data_prediction = lin_reg_model.predict(X_train)  # prediction on Training data

error_score = metrics.r2_score(Y_train, training_data_prediction)  # R squared Error

    # Visualize the Actual prices and Predicted Prices
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

    # Lasso Regression

lass_reg_model = Lasso()  # loading the linear regression model
lass_reg_model.fit(X_train, Y_train)

    #  Model Evaluation 2
test_data_prediction = lass_reg_model.predict(X_test)  # prediction on Training data
error_score = metrics.r2_score(Y_test, test_data_prediction)  # R squared Error


plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()