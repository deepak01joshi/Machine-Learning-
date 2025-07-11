# 1. What is Simple Linear Regression?
"""
Simple Linear Regression is used to model the relationship between two variables by fitting a straight line.
Equation: Y = mX + c
"""

# 2. Key assumptions of Simple Linear Regression
"""
- Linearity: Y and X have a linear relationship
- Independence: Observations are independent
- Homoscedasticity: Constant variance of residuals
- Normality: Residuals are normally distributed
"""

# 3. Meaning of coefficient m in Y = mX + c
"""
m is the slope. It tells how much Y changes for each unit increase in X.
"""

# 4. Meaning of intercept c in Y = mX + c
"""
c is the intercept. It is the value of Y when X = 0.
"""

# 5. How is the slope m calculated?
"""
m = sum((Xi - mean(X)) * (Yi - mean(Y))) / sum((Xi - mean(X))^2)
"""

# 6. Purpose of least squares method
"""
To minimize the sum of the squared differences between actual Y and predicted Y.
"""

# 7. Interpretation of R² (coefficient of determination)
"""
R² tells how much of the variance in Y is explained by X. R² = 1 is perfect; R² = 0 means no relation.
"""

# ========================================
# MULTIPLE LINEAR REGRESSION
# ========================================

# 8. What is Multiple Linear Regression?
"""
A method to predict Y using multiple predictors (X1, X2, ..., Xn).
Y = b0 + b1*X1 + b2*X2 + ... + bn*Xn
"""

# 9. Main difference: Simple vs. Multiple
"""
Simple: 1 predictor; Multiple: 2 or more predictors.
"""

# 10. Assumptions of Multiple Linear Regression
"""
Same as Simple Linear Regression +
- No multicollinearity: Predictors shouldn’t be highly correlated.
"""

# 11. What is heteroscedasticity?
"""
It means error variance changes with X. It makes standard errors unreliable.
"""

# 12. How to improve model with high multicollinearity?
"""
- Remove or combine correlated variables
- Use Ridge/Lasso regression
- Use PCA
"""

# 13. Transforming categorical variables
"""
- One-hot encoding
- Label encoding
- Target encoding
"""

# 14. Role of interaction terms
"""
They model the combined effect of two or more variables on Y.
"""

# 15. Interpretation of intercept (Simple vs Multiple)
"""
Simple: Y when X = 0
Multiple: Y when all Xs = 0 (might not be practical)
"""

# 16. Significance of slope
"""
Tells how much Y changes with a 1-unit increase in X (keeping other variables constant).
"""

# 17. Context of intercept
"""
Gives baseline value of Y when all inputs are 0. Interpretation depends on context.
"""

# 18. Limitations of R²
"""
- Doesn’t show if the model will predict well
- Can increase just by adding variables
- Doesn’t show causation
"""

# ========================================
# ADVANCED & POLYNOMIAL REGRESSION
# ========================================

# 19. Interpretation of a large standard error for a coefficient
"""
Means the estimate of the coefficient is unstable. It may not be statistically significant.
"""

# 20. Identifying heteroscedasticity in residual plots
"""
Look for a funnel shape in residual vs. fitted plots. It's important to address because it affects reliability.
"""

# 21. High R² but low adjusted R²
"""
Suggests that some predictors may be irrelevant and are not improving the model.
"""

# 22. Importance of scaling variables
"""
Ensures all variables are on the same scale, improving model performance and interpretation.
"""

# 23. What is polynomial regression?
"""
A form of regression where the relationship between X and Y is modeled as an nth-degree polynomial.
"""

# 24. How is it different from linear regression?
"""
Linear: straight line
Polynomial: curved line (e.g., quadratic, cubic)
"""

# 25. When is it used?
"""
When the data shows a nonlinear trend.
"""

# 26. General equation for polynomial regression
"""
Y = b0 + b1*X + b2*X^2 + ... + bn*X^n
"""

# 27. Can polynomial regression be used for multiple variables?
"""
Yes. You can include polynomial terms for multiple variables.
"""

# 28. Limitations of polynomial regression
"""
- Prone to overfitting
- Becomes unstable at high degrees
- Harder to interpret
"""

# 29. Evaluating polynomial degree
"""
- Use cross-validation
- Check metrics like RMSE, R², and Adjusted R²
"""

# 30. Importance of visualization
"""
Helps see the shape of the curve and if the model fits the data well.
"""

# 31. Implementing polynomial regression in Python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])

# Polynomial regression (degree 2)
model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(X, y)
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Polynomial Fit')
plt.legend()
plt.title('Polynomial Regression Example')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
