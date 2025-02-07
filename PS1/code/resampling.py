'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self, min_particles=100, max_particles=2000, ess_threshold=0.5, multiplier = 0.01):
        """
        TODO : Initialize resampling process parameters here
        """
        self.min_particles = min_particles
        self.max_particles = max_particles
        self.ess_threshold = ess_threshold
        self.increase_multiplier = 1.0 + multiplier
        self.decrease_multiplier = 1.0 - multiplier

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        num_particles = X_bar.shape[0]
        weights = X_bar[:, 3]
        weights /= np.sum(weights)

        indices = np.random.choice(num_particles, size=num_particles, p=weights)
        X_bar_resampled = X_bar[indices]
        X_bar_resampled[:, 3] = 1.0 / num_particles

        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        #if np.sum(X_bar[:,3]) == 0:
            #X_bar[:,3] = 1/X_bar.shape[0]

        # Normalize weights
        X_bar[:,3] /= np.sum(X_bar[:,3])
        X_bar_resampled =  np.zeros_like(X_bar)

        # Low variance sampler from Probabilistic robotics Chapter 4.3
        r = np.random.uniform(0, 1/(X_bar.shape[0]))
        c = X_bar[0][3]
        i = 0
        for m in range(X_bar.shape[0]):
            U = r + (m)/X_bar.shape[0]
            while U > c:
                i += 1
                c += X_bar[i][3]
            X_bar_resampled[m] = X_bar[i]
            X_bar_resampled[m][3] = 1/X_bar.shape[0]
        return X_bar_resampled

    def dynamic_low_variance_sampler(self, X_bar):
        """
        Implements low variance resampling with dynamic particle adjustment based on ESS.
        """

        num_particles = X_bar.shape[0]
        weights = X_bar[:, 3]
        weights /= np.sum(weights)

        # Compute ESS ratio
        ess = self.effective_sample_size(weights)
        ess_ratio = ess / num_particles

        # Adjust particle count
        if ess_ratio < self.ess_threshold:
            num_particles = min(int(num_particles * self.increase_multiplier), self.max_particles)
        elif ess_ratio > (1 - self.ess_threshold):
            num_particles = max(int(num_particles * self.decrease_multiplier), self.min_particles)

        # Low variance resampling
        X_bar_resampled = np.zeros((num_particles, 4))
        r = np.random.uniform(0, 1 / num_particles)
        c = weights[0]
        i = 0

        for m in range(num_particles):
            U = r + m / num_particles
            while U > c:
                i += 1
                c += weights[i]
            X_bar_resampled[m] = X_bar[i]
            X_bar_resampled[m][3] = 1.0 / num_particles

        print(f"Adjusted Number of Particles: {num_particles}, ESS: {ess_ratio}")

        return X_bar_resampled

    def dynamic_multinomial_sampler(self, X_bar):
        """
        Implements multinomial resampling with dynamic particle adjustment based on ESS.
        """

        num_particles = X_bar.shape[0]
        weights = X_bar[:, 3]
        weights /= np.sum(weights)

        # Compute ESS ratio
        ess = self.effective_sample_size(weights)
        ess_ratio = ess / num_particles

        # Adjust particle count
        if ess_ratio < self.ess_threshold:
            num_particles = min(int(num_particles * self.increase_multiplier), self.max_particles)
        elif ess_ratio > (1 - self.ess_threshold):
            num_particles = max(int(num_particles * self.decrease_multiplier), self.min_particles)

        # Multinomial resampling
        indices = np.random.choice(X_bar.shape[0], size=num_particles, p=weights)
        X_bar_resampled = X_bar[indices]
        X_bar_resampled[:, 3] = 1.0 / num_particles

        print(f"Adjusted Number of Particles: {num_particles}, ESS: {ess_ratio}")

        return X_bar_resampled
    
    def effective_sample_size(self, weights):
        return 1.0 / np.sum(weights**2)