import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost_func_logreg(theta, X, y):
    m = y.size
    J = 0
    grad_J = np.zeros(theta.shape)
    Z = np.dot(X, theta)
    h_theta = sigmoid(Z)
    J = -1 / m * (np.dot(y.T, np.log(h_theta)) + np.dot((1 - y).T, np.log(1 - h_theta)))
    grad_J = 1 / m * np.dot(X.T, h_theta - y)
    return J, grad_J

def grad_cost_logreg(X, y, theta, alpha, num_iters):
    J_iter = np.zeros(num_iters)
    for iter in range(num_iters):
        J_iter[iter], grad = cost_func_logreg(theta, X, y)
        theta = theta - alpha * grad

    return theta, J_iter

def plot_logreg_line(X, y, theta):
    """
        plot_reg_line plots the data points and regression line
        for linear regression
        Input arguments: X - np array (m, n) - independent variable.
        y - np array (m,1) - target variable
        theta - parameters
    """

    ind = 1
    x1_min = 1.1*X[:, ind].min()
    x1_max = 1.1*X[ :,ind].max()
    x2_min = -(theta[0] + theta[1]*x1_min)/theta[2]
    x2_max = -(theta[0] + theta[1]*x1_max)/theta[2]
    x1 = X
    x1lh = np.array([x1_min, x1_max])
    x2lh = np.array([x2_min, x2_max])
    x1 = X[: , 1]
    x2 = X[: , 2]
    plt.plot()
    plt. plot(x1[y[:,0] == 0], x2[y[:,0] == 0], 'rd', x1[y[:, 0] == 1], x2[y[:,0] == 1],
    'go', x1lh, x2lh, 'b-')
    plt.xlabel('exam1')
    plt.ylabel('exam2')
    plt.title('train data for admittance classifier')
    plt.grid()
    plt.show()

def plot_quad_reg_line(X_orig, y, theta):
    """
    Plots the data points and quadratic decision boundary:
        z = θ0 + θ1*x1 + θ2*x2 + θ3*x1^2

    X_orig: (m, 2) original feature matrix (without bias or nonlinear terms)
    y: (m, 1) binary labels
    theta: (4, 1) logistic regression parameters
    """
    # Extract original features
    x1 = X_orig[:, 0]
    x2 = X_orig[:, 1]

    # Plot the data points
    plt.scatter(x1[y[:, 0] == 0], x2[y[:, 0] == 0], c='red', marker='d', label='Class 0')
    plt.scatter(x1[y[:, 0] == 1], x2[y[:, 0] == 1], c='green', marker='o', label='Class 1')

    # Create a grid of values
    x1_vals = np.linspace(x1.min() - 1, x1.max() + 1, 200)
    x2_vals = np.linspace(x2.min() - 1, x2.max() + 1, 200)
    xx1, xx2 = np.meshgrid(x1_vals, x2_vals)

    # Compute z over the grid
    z = (theta[0] +
         theta[1] * xx1 +
         theta[2] * xx2 +
         theta[3] * (xx1 ** 2))

    # Plot contour where z = 0 (decision boundary)
    plt.contour(xx1, xx2, z, levels=[0], colors='blue', linewidths=2)

    plt.xlabel('First feature')
    plt.ylabel('Second feature')
    plt.grid(True)
    plt.show()
    return

def predict(X, theta):
  probs = sigmoid(np.dot(X, theta))
  return (probs >= 0.5).astype(int)

df = pd.read_csv('SVMtrain.csv')
df.head()

df = df.drop(columns=['PassengerId'])

df['Sex'] = df['Sex'].map({'Male' : 0, 'female' : 1})
df['Embarked'] = df['Embarked'].astype(str).map({'1' : 0, '2' : 1, '3' : 2})
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

X = df.drop('Survived', axis=1).values
y = df['Survived'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

theta_init = np.zeros((X_train.shape[1], 1))
alpha = 0.001
num_iters = 150000
theta, J_history = grad_cost_logreg(X_train, y_train, theta_init, alpha, num_iters)

y_pred = predict(X_test, theta)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function over Iterations")
plt.grid(True)
plt.show()

<<<<<<< HEAD
# Example user input:
# Pclass=3, Sex=male, Age=22, SibSp=0, Parch=0, Fare=7.25, Embarked=3
user_raw = np.array([[3, 0, 22, 0, 0, 7.25, 2]])  # mapped 'Sex' and 'Embarked'

# Add bias term
user_input = np.hstack([np.ones((user_raw.shape[0], 1)), user_raw])

# Predict probability and class
prob = sigmoid(user_input @ theta)[0][0]
prediction = int(prob >= 0.5)

print(f"\nUser Input Prediction:")
print(f"Survival probability: {prob:.2f}")
print("Prediction:", "Survived" if prediction else "Did not survive")
