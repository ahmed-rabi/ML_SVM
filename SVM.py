import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import cvxopt
cvxopt.solvers.options['show_progress'] = False

# Step 1: Define the linear kernel
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

# Step 2: Solve the dual optimization problem using CVXOPT
def solve_dual_svm(X, y, C=1.0):
    n_samples = X.shape[0]
    
    # Compute the kernel matrix
    K = np.array([[linear_kernel(X[i], X[j]) for j in range(n_samples)] for i in range(n_samples)])
    
    # Convert numpy arrays to cvxopt matrices with correct data type
    P = cvxopt.matrix(np.outer(y, y) * K, tc='d')
    q = cvxopt.matrix(-np.ones((n_samples, 1)), tc='d')
    
    # Inequality constraints: 0 <= alpha <= C
    G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))), tc='d')
    h = cvxopt.matrix(np.hstack((np.zeros(n_samples), C * np.ones(n_samples))), tc='d')
    
    # Equality constraint: sum(alpha_i * y_i) = 0
    A = cvxopt.matrix(y.reshape(1, -1).astype(np.double), tc='d')
    b = cvxopt.matrix(np.zeros(1), tc='d')
    
    # Solve the quadratic programming problem
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    
    # Extract the optimal alpha values
    alpha = np.array(solution['x']).flatten()
    
    return alpha

# Step 3: Compute w and b from the optimal alphas
def compute_parameters(X, y, alpha, threshold=1e-5):
    # Compute w using the formula: w = sum(alpha_i * y_i * x_i)
    support_mask = alpha > threshold
    support_vectors = X[support_mask]
    support_alphas = alpha[support_mask]
    support_labels = y[support_mask]
    
    w = np.sum(support_alphas.reshape(-1, 1) * support_labels.reshape(-1, 1) * support_vectors, axis=0)
    
    # Compute b using KKT conditions
    # We use the first support vector and the fact that for support vectors:
    # y_i(w^T x_i + b) = 1
    x_sv = support_vectors[0]
    y_sv = support_labels[0]
    b = y_sv - np.dot(w, x_sv)
    
    return w, b

# Step 4: Define the SVM class
class SVM:
    def __init__(self, C=1.0):
        self.C = C
        self.w = None
        self.b = None
        self.alpha = None
    
    def fit(self, X, y):
        # Convert labels to {-1, 1}
        y = np.where(y <= 0, -1, 1)
        
        # Solve the dual optimization problem
        self.alpha = solve_dual_svm(X, y, self.C)
        
        # Compute w and b
        self.w, self.b = compute_parameters(X, y, self.alpha)
        
        return self
    
    def predict(self, X):
        # Make predictions using w and b
        scores = np.dot(X, self.w) + self.b
        return np.sign(scores)

# Example usage
X, y = make_classification(
    n_samples=2600,          # Total number of samples
    n_features=28*28,        # Number of features (28x28 grayscale image)
    n_informative=2,         # Number of informative features
    n_redundant=0,           # No redundant features
    n_clusters_per_class=1,  # Single cluster per class
    class_sep=2.0,           # Separation between classes
    random_state=42          # For reproducibility
)

# # Add Gaussian noise to the data
# noise_level = 0.5  # Adjust this value to control noise intensity (0.0 to 1.0)
# noise = np.random.normal(0, noise_level, X.shape)
# X = X + noise

# # Clip values to ensure they stay in a valid range [-1, 1]
# X = np.clip(X, -1, 1)

# convert labels to -1 and 1
y = 2 * y - 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
svm = SVM(C=1.0)
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)

# Print results
print("Support vectors:", svm.alpha > 1e-5)
print("Weights:", svm.w)
print("Bias:", svm.b)
print("Predictions:", y_pred)
print("Accuracy:", np.mean(y_pred == y_test) * 100, "%")
