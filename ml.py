import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing(as_frame=True)
housing = housing.frame

X = housing.drop("MedHouseVal", axis=1)
y = housing["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
gbr = GradientBoostingRegressor(loss='quantile')
param_grid = {
    'alpha': [0.1, 0.25, 0.5, 0.75, 0.9]
}
grid = GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    cv=5,
    scoring='r2'
)

grid.fit(X_train, y_train)
print("Best Alpha:", grid.best_params_)
print("Best Score:", grid.best_score_)
gbr = GradientBoostingRegressor(alpha=0.5,max_depth=5)

gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)