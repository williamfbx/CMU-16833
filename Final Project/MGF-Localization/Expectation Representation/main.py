import math
import matplotlib.pyplot as plt
import numpy as np

def compute_moment_order_1_and_2(alpha, A, B, u_moments, Z_moments, X_prev_moments):
    E1 = 0.0  # E[X_t]
    E2 = 0.0  # E[X_t^2]

    # n = 1
    for i in range(2):  # i = 0, 1
        for j in range(2 - i):  # j = 0, 1 if i=0; j = 0 if i=1
            k = 1 - i - j
            coeff = 1
            b = ((1 - alpha) ** i) * (B ** i) * u_moments.get(i, 0.0)
            z = (alpha ** j) * Z_moments.get(j, 0.0)
            x = ((1 - alpha) ** k) * (A ** k) * X_prev_moments.get(k, 0.0)
            E1 += coeff * b * z * x

    # n = 2
    for i in range(3):  # i = 0,1,2
        for j in range(3 - i):  # j = 0,1,2
            k = 2 - i - j
            coeff = math.factorial(2) / (math.factorial(i) * math.factorial(j) * math.factorial(k))
            b = ((1 - alpha) ** i) * (B ** i) * u_moments.get(i, 0.0)
            z = (alpha ** j) * Z_moments.get(j, 0.0)
            x = ((1 - alpha) ** k) * (A ** k) * X_prev_moments.get(k, 0.0)
            E2 += coeff * b * z * x

    return {1: E1, 2: E2}

def visualize_distribution(mu, sigma2, title="Distribution"):
    if sigma2 <= 0:
        sigma2 = 1e-6  # prevent divide by zero or negative variance

    x = np.linspace(mu - 3 * np.sqrt(sigma2), mu + 3 * np.sqrt(sigma2), 500)
    y = (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-0.5 * ((x - mu) ** 2) / sigma2)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label=f'μ={mu:.2f}, σ²={sigma2:.4f}')
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.show()

def parse_input(input_lines):
    commands = []
    for line in input_lines:
        parts = line.strip().split()
        if parts[0] == "C":
            commands.append(("C", float(parts[1])))
        elif parts[0] == "L":
            commands.append(("L", float(parts[1]), float(parts[2])))
    return commands

def main():
    alpha = 0.5
    A = 1.0
    B = 1.0

    # Initial state moments (assume small uncertainty)
    X_moments = {0: 1.0, 1: 0.0, 2: 0.01}
    u_current = 0.0
    Z_current = None

    input_lines = [
        "L 0 0",
        "C 1",
        "C 1",
        "C 1",
        "C 1",
        "C 1",
        "L 0 0.2",
        "C 1",
        "C 1",
        "C 1",
        "C 1",
        "C 1",
        "L 0 -0.5"
    ]

    commands = parse_input(input_lines)

    for step, cmd in enumerate(commands):
        if cmd[0] == "C":
            u_current = cmd[1]
        elif cmd[0] == "L":
            Z_current = cmd[2]

        # Always perform propagation
        u_moments = {0: 1.0, 1: u_current, 2: u_current ** 2}
        if Z_current is not None:
            Z_moments = {0: 1.0, 1: Z_current, 2: Z_current ** 2}
        else:
            Z_moments = {0: 1.0, 1: 0.0, 2: 0.0}

        X_moments = compute_moment_order_1_and_2(alpha, A, B, u_moments, Z_moments, X_moments)

        mu = X_moments[1]
        sigma2 = X_moments[2] - mu ** 2
        print(f"Step {step}: mean = {mu:.4f}, variance = {sigma2:.4f}")
        visualize_distribution(mu, sigma2, title=f"Step {step}")

if __name__ == "__main__":
    main()
