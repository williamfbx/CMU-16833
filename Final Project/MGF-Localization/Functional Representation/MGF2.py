import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sympy import symbols, diff, exp, integrate, oo, sqrt, pi, simplify, N
import math

# --------------------------
# 1D MGF Localizer (for each coordinate)
# --------------------------
class MGFlocalizer:
    def __init__(self, A, B, alpha):
        self.A = A
        self.B = B
        self.alpha = alpha
        self.s = symbols('s')
        self.x = symbols('x')
        
    def get_MGF_from_pdf(self, pdf):
        try:
            t = symbols('t', real=True)
            pdf_t = pdf.subs(self.x, t)
            mgf = integrate(exp(self.s * t) * pdf_t, (t, -oo, oo)).doit().simplify()
            return mgf
        except Exception as e:
            raise ValueError(f"Failed to compute MGF from PDF: {e}")
    
    def propagate_MGF(self, MX_prev, MZ, Mu):
        a, A, B, s = self.alpha, self.A, self.B, self.s
        return simplify(MZ.subs(s, a*s) * Mu.subs(s, (1-a)*B*s) * MX_prev.subs(s, (1-a)*A*s))
    
    def compute_moments(self, MGF, order=2):
        moments = {}
        s = self.s
        MGF_numeric = N(MGF)
        for n in range(1, order+1):
            derivative = diff(MGF_numeric, s, n)
            derivative_value = derivative.subs(s, 0)
            derivative_value = N(derivative_value)
            try:
                moments[n] = float(derivative_value)
            except Exception as e:
                raise ValueError(f"Invalid moment at order {n}: {derivative_value}, error: {e}")
        return moments

# --------------------------
# Environment Setup
# --------------------------
# Define dimensions:
corridor_width = 5            
east_corridor_length = 10    
south_corridor_length = 20   
room_size = 20                

def sample_line(p0, p1, num_points=5):
    pts = []
    for t in np.linspace(0, 1, num_points):
        pts.append(p0 + t*(p1 - p0))
    return pts

landmarks = []

# East corridor walls 
east_top = sample_line(np.array([0, corridor_width/2]), np.array([east_corridor_length, corridor_width/2]))
east_bottom = sample_line(np.array([0, -corridor_width/2]), np.array([east_corridor_length, -corridor_width/2]))
landmarks += [tuple(pt) for pt in east_top + east_bottom]

# South corridor walls 
south_left = sample_line(np.array([7.5, 0]), np.array([7.5, -south_corridor_length]))
south_right = sample_line(np.array([12.5, 0]), np.array([12.5, -south_corridor_length]))
landmarks += [tuple(pt) for pt in south_left + south_right]

# Room walls
room_origin = np.array([10, -south_corridor_length - room_size])
room_north = sample_line(np.array([10, -south_corridor_length]), np.array([10+room_size, -south_corridor_length]))
room_south = sample_line(np.array([10, -south_corridor_length - room_size]), np.array([10+room_size, -south_corridor_length - room_size]))
room_west = sample_line(np.array([10, -south_corridor_length - room_size]), np.array([10, -south_corridor_length]))
room_east = sample_line(np.array([10+room_size, -south_corridor_length - room_size]), np.array([10+room_size, -south_corridor_length]))
landmarks += [tuple(pt) for pt in room_north + room_south + room_west + room_east]

# Sensor parameters
sensor_range = 10.0  
sigma_r = 0.2        
sigma_theta = 0.1    
sigma_sensor = 0.2   

# --------------------------
# New Sensor Simulation: 6 discrete sensor readings per time step.
# --------------------------
def simulate_sensor_readings(true_state, landmarks):
    sensor_relative_angles = [0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3]
    x_r, y_r, heading = true_state
    readings = []
    threshold = math.pi/6  
    for a in sensor_relative_angles:
        global_angle = heading + a
        best_distance = sensor_range + 1  
        best_landmark = None
        for L in landmarks:
            Lx, Ly = L
            dx = Lx - x_r
            dy = Ly - y_r
            dist = math.hypot(dx, dy)
            if dist <= sensor_range:
                bearing = math.atan2(dy, dx)
                angle_diff = abs((bearing - global_angle + math.pi) % (2*math.pi) - math.pi)
                if angle_diff < threshold and dist < best_distance:
                    best_distance = dist
                    best_landmark = L
        if best_landmark is not None:
            # Simulate measurement with noise.
            r_true = best_distance
            r_meas = r_true + np.random.normal(0, sigma_r)
            a_meas = a + np.random.normal(0, sigma_theta)
            global_angle_meas = heading + a_meas
            Lx, Ly = best_landmark
            # Invert the measurement: estimated robot position = landmark position - measured offset.
            est_x = Lx - r_meas * math.cos(global_angle_meas)
            est_y = Ly - r_meas * math.sin(global_angle_meas)
        else:
            # No landmark found: return max range reading.
            r_meas = sensor_range + np.random.normal(0, sigma_r)
            a_meas = a + np.random.normal(0, sigma_theta)
            global_angle_meas = heading + a_meas
            Lx = x_r + sensor_range * math.cos(global_angle)
            Ly = y_r + sensor_range * math.sin(global_angle)
            est_x = Lx - r_meas * math.cos(global_angle_meas)
            est_y = Ly - r_meas * math.sin(global_angle_meas)
        readings.append((est_x, est_y))
    return readings

# --------------------------
# Robot Path Planning (Phases)
# --------------------------
def interpolate_path(start, end, steps, fixed_heading=None, heading_start=None, heading_end=None):
    path = []
    for i in range(steps):
        t_ratio = i / (steps - 1) if steps > 1 else 0
        pos = start + t_ratio*(end - start)
        if fixed_heading is not None:
            heading = fixed_heading
        else:
            heading = heading_start + t_ratio*(heading_end - heading_start)
        path.append(np.array([pos[0], pos[1], heading]))
    return path

phase1 = interpolate_path(np.array([0, 0]), np.array([east_corridor_length, 0]), 10, fixed_heading=0.0)
phase2 = interpolate_path(np.array([east_corridor_length, 0]), np.array([east_corridor_length, 0]), 5, heading_start=0.0, heading_end=-np.pi/2)
phase3 = interpolate_path(np.array([east_corridor_length, 0]), np.array([east_corridor_length, -south_corridor_length]), 20, fixed_heading=-np.pi/2)
phase4 = interpolate_path(np.array([east_corridor_length, -south_corridor_length]), np.array([east_corridor_length, -south_corridor_length]), 5, heading_start=-np.pi/2, heading_end=0.0)
phase5 = interpolate_path(np.array([east_corridor_length, -south_corridor_length]), np.array([east_corridor_length+room_size, -south_corridor_length]), 20, fixed_heading=0.0)

true_states = phase1 + phase2 + phase3 + phase4 + phase5

# Optionally, add a small random noise to the true states.
motion_noise_std = 0.05
for i in range(len(true_states)):
    noise = np.random.normal(0, motion_noise_std, 2)
    true_states[i][0] += noise[0]
    true_states[i][1] += noise[1]

# --------------------------
# MGF SLAM Setup (for x and y)
# --------------------------
A, B, alpha = 1.0, 1.0, 0.6
localizer_x = MGFlocalizer(A, B, alpha)
localizer_y = MGFlocalizer(A, B, alpha)
sigma_u2 = 0.05  # control noise variance

MX = None
MY = None
estimates = [] 

# --------------------------
# Main SLAM Simulation Loop
# --------------------------
for t in range(1, len(true_states)):
    print(f"Time step {t}:")
    current_state = true_states[t]
    # Simulate sensor readings: exactly 6 sensor readings.
    sensor_readings = simulate_sensor_readings(current_state, landmarks)
    sensor_estimates_x = [est[0] for est in sensor_readings]
    sensor_estimates_y = [est[1] for est in sensor_readings]
    
    # Build sensor PDFs for x and y.
    from sympy import sqrt, pi
    sensor_pdf_x = 0
    sensor_pdf_y = 0
    n = len(sensor_readings)
    for est in sensor_estimates_x:
        sensor_pdf_x += (1/n) * (1/(sqrt(2*pi)*sigma_sensor)) * exp(-((localizer_x.x - est)**2)/(2*sigma_sensor**2))
    for est in sensor_estimates_y:
        sensor_pdf_y += (1/n) * (1/(sqrt(2*pi)*sigma_sensor)) * exp(-((localizer_y.x - est)**2)/(2*sigma_sensor**2))
    
    # Compute sensor MGFs.
    MZ_x = localizer_x.get_MGF_from_pdf(sensor_pdf_x)
    MZ_y = localizer_y.get_MGF_from_pdf(sensor_pdf_y)
    
    # Control input: based on displacement between consecutive true states.
    dx = true_states[t][0] - true_states[t-1][0]
    dy = true_states[t][1] - true_states[t-1][1]
    Mu_x = exp(dx * localizer_x.s + 0.5 * sigma_u2 * localizer_x.s**2)
    Mu_y = exp(dy * localizer_y.s + 0.5 * sigma_u2 * localizer_y.s**2)
    
    if t == 1:
        MX = MZ_x
        MY = MZ_y
    else:
        MX = localizer_x.propagate_MGF(MX, MZ_x, Mu_x)
        MY = localizer_y.propagate_MGF(MY, MZ_y, Mu_y)
    
    moments_x = localizer_x.compute_moments(MX, order=2)
    moments_y = localizer_y.compute_moments(MY, order=2)
    mean_x = moments_x[1]
    var_x = moments_x[2] - moments_x[1]**2
    mean_y = moments_y[1]
    var_y = moments_y[2] - moments_y[1]**2
    estimates.append((mean_x, mean_y, var_x, var_y))

# --------------------------
# Plotting the Environment and Trajectories
# --------------------------
def plot_environment_and_path(true_states, estimates):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.add_patch(Rectangle((0, -corridor_width/2), east_corridor_length, corridor_width, edgecolor='black', facecolor='none'))
    ax.add_patch(Rectangle((east_corridor_length - corridor_width/2, -south_corridor_length), corridor_width, south_corridor_length, edgecolor='black', facecolor='none'))
    ax.add_patch(Rectangle((east_corridor_length, -south_corridor_length - room_size), room_size, room_size, edgecolor='black', facecolor='none'))
    
    # Plot landmarks.
    lm = np.array(landmarks)
    if lm.size > 0:
        ax.plot(lm[:,0], lm[:,1], 'rx', label='Landmarks')
    
    # Plot true path.
    true_path = np.array([[s[0], s[1]] for s in true_states])
    ax.plot(true_path[:,0], true_path[:,1], 'g--', label='True Path')
    
    # Plot estimated path with error bars.
    if estimates:
        est_path = np.array([[est[0], est[1]] for est in estimates])
        x_err = np.sqrt([est[2] for est in estimates])
        y_err = np.sqrt([est[3] for est in estimates])
        ax.errorbar(est_path[:,0], est_path[:,1], xerr=x_err, yerr=y_err, fmt='bo-', capsize=3, label='Estimated Path')
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('SLAM Simulation: East then South into Room with 6 Sensor Readings per Time Step')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    plt.show()

plot_environment_and_path(true_states, estimates)
