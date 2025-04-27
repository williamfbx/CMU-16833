import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, exp, lambdify, integrate, oo, sqrt, pi

class MGFlocalizer:
    def __init__(self, A, B, alpha):
        self.A = A
        self.B = B
        self.alpha = alpha
        self.s, self.x = symbols('s x')

    def get_MGF_from_pdf(self, pdf):
        try:
            mgf = integrate(exp(self.s * self.x) * pdf, (self.x, -oo, oo)).simplify()
            return mgf
        except Exception as e:
            raise ValueError(f"Failed to compute MGF from PDF: {e}")

    def propagate_MGF(self, MX_prev, MZ, Mu):
        a, A, B, s = self.alpha, self.A, self.B, self.s
        return (MZ.subs(s, a * s) * Mu.subs(s, (1 - a) * B * s) * MX_prev.subs(s, (1 - a) * A * s)).simplify()

    def compute_moments(self, MGF, order=2):
        moments = {}
        s = self.s
        for n in range(1, order + 1):
            moment_expr = diff(MGF, s, n).subs(s, 0).evalf()
            if moment_expr.is_real and not moment_expr.has(np.nan):
                moments[n] = float(moment_expr)
            else:
                raise ValueError(f"Invalid moment at order {n}: {moment_expr}")
        return moments

# Parameters
A, B, alpha, T = 1.0, 1.0, 0.6, 20
sensor_range = 8
localizer = MGFlocalizer(A, B, alpha)

x = localizer.x
mu_u, sigma_u2 = 1.0, 0.05
Mu = exp(mu_u * localizer.s + 0.5 * sigma_u2 * localizer.s**2)

obstacles = [5, 10, 15]
true_position = 0
means, variances, true_positions = [], [], []
MX = None

for t in range(T):
    true_position += 1
    true_positions.append(true_position)

    nearby_obs = [obs for obs in obstacles if obs - true_position > 0 and obs - true_position <= sensor_range]
    if nearby_obs:
        sensor_pdf = sum((1 / len(nearby_obs)) * (1 / sqrt(2 * pi * 0.2)) * exp(-((x - obs)**2) / (2 * 0.2)) for obs in nearby_obs)
    else:
        sensor_pdf = (1 / sqrt(2 * pi * 0.05)) * exp(-((x - true_position)**2) / (2 * 0.05))

    try:
        MZ = localizer.get_MGF_from_pdf(sensor_pdf)

        if t == 0:
            MX = MZ  # Initialize belief from first sensor reading
        else:
            MX = localizer.propagate_MGF(MX, MZ, Mu)

        moments = localizer.compute_moments(MX, order=2)
        means.append(moments[1])
        variances.append(moments[2] - moments[1]**2)
    except Exception as e:
        print(f"Error at timestep {t}: {e}")
        break

# Plot results
plt.figure(figsize=(10, 5))
steps = np.arange(1, len(means) + 1)

for obs in obstacles:
    plt.axhline(obs, color='red', linestyle='--', label='Obstacle' if obs == obstacles[0] else "")

plt.errorbar(steps, means, yerr=np.sqrt(variances), fmt='o-', capsize=5, label='Estimated position')
plt.plot(steps, true_positions[:len(means)], 'g--', label='True position')

plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('MGF Localization with Dynamic Sensor PDFs and Obstacle-aware Updates')
plt.legend()
plt.grid()
plt.show()