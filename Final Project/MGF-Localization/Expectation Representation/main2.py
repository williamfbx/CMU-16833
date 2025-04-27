# Corrected MGF Localization based exactly on provided formulas
import numpy as np
import matplotlib.pyplot as plt

def mgf_update(X_prev_mean, X_prev_var, u, z, alpha=0.5, A=1.0, B=1.0):
    # Mean update
    X_mean = alpha * z + (1 - alpha) * (A * X_prev_mean + B * u)
    
    # Variance update (exactly as in provided formula)
    var_Z = 0.04  # assume sensor variance fixed (e.g., sigma_z^2 = 0.2^2)
    X_var = (alpha ** 2) * var_Z + ((1 - alpha) ** 2) * (A**2 * X_prev_var)
    
    return X_mean, X_var

def main():
    alpha = 0.3  # Tune alpha to rely more on the motion model initially
    A = 1.0
    B = 1.0
    u = 1.0

    true_positions = [0]
    estimated_means = [0]
    estimated_vars = [1.0]

    for step in range(1, 21):
        # True position update
        true_pos = true_positions[-1] + u
        true_positions.append(true_pos)
        
        # Measurement (landmark at 0): noisy observation
        measurement_noise = np.random.normal(0, 0.2)
        z = true_pos + measurement_noise

        # MGF Update (exact formula from provided image)
        mean, var = mgf_update(estimated_means[-1], estimated_vars[-1], u, z, alpha, A, B)
        
        estimated_means.append(mean)
        estimated_vars.append(var)

    steps = np.arange(len(estimated_means))
    obstacles = [5, 10, 15]

    plt.figure(figsize=(10, 5))
    for obs in obstacles:
        plt.axhline(obs, color='red', linestyle='--', label='Obstacle' if obs == obstacles[0] else "")

    plt.errorbar(steps, estimated_means, yerr=np.sqrt(estimated_vars), fmt='o-', capsize=5, label='Estimated position')
    plt.plot(steps, true_positions, 'g--', label='True position')

    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.title('MGF Localization with Correct Mean and Variance Formulas')
    plt.legend()
    plt.grid()
    plt.show()

main()
