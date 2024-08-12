import numpy as np

# Initialize parameters
F = np.array([[1, 1], [0, 1]])  # State transition matrix (constant velocity model)
B = np.array([[0], [0]])       # Control input matrix (none in this case)
H = np.array([[1, 0]])         # Measurement matrix (position measurement)
Q = np.eye(2) * 0.01           # Process noise covariance (adjust as needed)
R = np.array([[0.1]])          # Measurement noise covariance (adjust as needed)

# Initial state and covariance
x = np.array([[0], [60]])      # Initial position and velocity
P = np.eye(2) * 100            # Initial state covariance

# Simulate measurements (replace with actual data)
def getMeasurement(updateNumber):
    dt = 0.1
    w = 8 * np.random.randn(1)
    v = 8 * np.random.randn(1)
    z = x[0] + x[1] * dt + v
    x[0] = z - v
    x[1] = 60 + w
    return z

# Kalman filter loop
for updateNumber in range(1, 101):  # Simulate 100 measurements
    z = getMeasurement(updateNumber)

    # Prediction step
    x_predicted = np.dot(F, x)
    P_predicted = np.dot(np.dot(F, P), F.T) + Q

    # Update step
    S = np.dot(np.dot(H, P_predicted), H.T) + R
    K = np.dot(np.dot(P_predicted, H.T), np.linalg.inv(S))
    y = z - np.dot(H, x_predicted)
    x = x_predicted + np.dot(K, y)
    I = np.eye(2)
    P = np.dot(np.dot(I - np.dot(K, H), P_predicted), (I - np.dot(K, H)).T) + np.dot(np.dot(K, R), K.T)

    print(f"Measurement {updateNumber}: Estimated position = {x[0][0]:.2f}, Estimated velocity = {x[1][0]:.2f}")

# Plot results or use the estimated state as needed
