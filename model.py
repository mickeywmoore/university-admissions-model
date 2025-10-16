import numpy as np

def sigmoid(z):
    """Computes the sigmoid function."""
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b):
    """Computes the log loss cost for logistic regression."""
    m = X.shape[0]
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    total_cost = (1/m) * np.sum(-y * np.log(f_wb) - (1 - y) * np.log(1 - f_wb))
    return total_cost

def compute_gradient(X, y, w, b):
    """Computes the gradient for logistic regression."""
    m = X.shape[0]
    f_wb = sigmoid(np.dot(X, w) + b)
    err = f_wb - y
    dj_dw = (1/m) * np.dot(X.T, err)
    dj_db = (1/m) * np.sum(err)
    return dj_db, dj_dw

def predict(X, w, b):
    """Predicts 0 or 1 using the learned parameters."""
    probabilities = sigmoid(np.dot(X, w) + b)
    return (probabilities >= 0.5).astype(int)

# --- This part of the script will run when you execute the file ---
if __name__ == '__main__':
    # This is a simple demonstration and not part of the core model logic.
    # It shows that the functions are defined and can be called.
    print("Logistic Regression Model functions are defined.")
    
    # Example usage with dummy data
    X_dummy = np.array([[1, 2], [3, 4]])
    y_dummy = np.array([0, 1])
    w_dummy = np.array([0.1, 0.2])
    b_dummy = -1
    
    cost = compute_cost(X_dummy, y_dummy, w_dummy, b_dummy)
    print(f"Example cost: {cost:.4f}")