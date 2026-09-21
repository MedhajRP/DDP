import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.stats import norm
import warnings



# Executes a standard, single-seed Bayesian Optimization loop on the Branin function using a custom Gaussian Process Regressor and Expected Improvement,
# demonstrating the optimization process under three different noise conditions (Gaussian, Student-T, and contaminated).

def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
    """Squared Exponential (RBF) Kernel"""
    sqdist = cdist(X1, X2, 'sqeuclidean')
    return variance * np.exp(-0.5 * (1/length_scale**2) * sqdist)

class CustomGPRegressor:
    """GP regression model that optimizes its own hyperparameters 
    using Marginal Likelihood, with y-normalization."""
    def __init__(self, noise_var=1e-4):
        self.noise_var = noise_var
        self.length_scale = 1.0
        self.variance = 1.0
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        
        # Track standardization parameters
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, X, y):
        self.X_train = np.atleast_2d(X)
        y_raw = np.atleast_1d(y).flatten()
        
        # Standardization
        self.y_mean = np.mean(y_raw)
        self.y_std = np.std(y_raw)
        
        # Prevent division by zero if all y-values are exactly the same
        if self.y_std < 1e-9:
            self.y_std = 1e-9 
            
        # Scale the data: (y-mean)/std
        self.y_train = (y_raw-self.y_mean)/self.y_std
        
        
        def objective(params):
            ls, var = params
            K = rbf_kernel(self.X_train, self.X_train, ls, var)+self.noise_var*np.eye(len(self.X_train))
            try:
                K_inv = np.linalg.inv(K)
                # The LML is calculated on the normalized y_train
                term1 = -0.5 * self.y_train.T.dot(K_inv).dot(self.y_train)
                term2 = -0.5 * np.linalg.slogdet(K)[1]
                term3 = -0.5 * len(self.X_train) * np.log(2 * np.pi)
                lml = term1 + term2 + term3
                return -lml 
            except np.linalg.LinAlgError:
                return 1e10

        initial_guess = [self.length_scale, self.variance]
        bounds = [(1e-3, 100.0), (1e-3, 100.0)]
        
        res = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        self.length_scale, self.variance = res.x
        
        K = rbf_kernel(self.X_train, self.X_train, self.length_scale, self.variance) + self.noise_var * np.eye(len(self.X_train))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_test, return_std=True):
        X_test = np.atleast_2d(X_test)
        K_s = rbf_kernel(self.X_train, X_test, self.length_scale, self.variance)
        
        # This mean is currently in the normalized space
        mu_norm = K_s.T.dot(self.K_inv).dot(self.y_train)
        
        if not return_std:
            return mu_norm * self.y_std + self.y_mean
            
        K_ss = rbf_kernel(X_test, X_test, self.length_scale, self.variance)
        cov = K_ss - K_s.T.dot(self.K_inv).dot(K_s)
        var_norm = np.maximum(np.diag(cov), 1e-9) 
        std_norm = np.sqrt(var_norm)
        
        
        # Scale predictions back to the original Branin/Camel elevation
        mu_actual = mu_norm * self.y_std + self.y_mean
        
        # Standard deviation scales directly with y_std (no mean addition for spread)
        std_actual = std_norm * self.y_std
        return mu_actual, std_actual


# 2. Benchmark Functions & Noise Generators

def branin(X):
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    return a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*np.cos(x1) + s

def generate_noise(n, noise_type='gaussian', scale=0.1):
    """Generates observation noise settings."""
    if noise_type == 'gaussian':
        return np.random.normal(0.0, scale, n)
    elif noise_type == 'student_t':
        return np.random.standard_t(df=2, size=n) * scale
    elif noise_type == 'contaminated':
        base = np.random.normal(0.0, scale, n)
        outliers = np.random.normal(0.0, scale * 20.0, n)
        mask = np.random.rand(n) < 0.10
        return np.where(mask, outliers, base)
    elif noise_type == 'none':
        return np.zeros(n)


# 3. Bayesian Optimization Loop Components

def expected_improvement(X, Y_sample, gp_model, xi=0.01):
    """Calculates Expected Improvement using GP model."""
    mu, sigma = gp_model.predict(X, return_std=True)
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    
    mu_opt = np.min(Y_sample)
    imp = mu_opt - mu - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 1e-9] = 0.0
    return ei.flatten()

def propose_location(X_sample, Y_sample, gp_model, bounds):
    """Optimizes acquisition function via random multi-start L-BFGS-B."""
    dim = bounds.shape[0]
    best_val = -np.inf
    best_x = None
    
    def min_obj(x):
        return -expected_improvement(x.reshape(1, -1), Y_sample, gp_model)[0]
    
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(10, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if -res.fun > best_val:
            best_val = -res.fun
            best_x = res.x
            
    return best_x.reshape(1, dim)

def run_bo_experiment(target_func, bounds, n_iters=15, noise_type='gaussian'):
    """Executes the BO loop."""
    print(f"Starting BO with {noise_type} noise...")
    
    # Initial random samples
    X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(5, bounds.shape[0]))
    noise = generate_noise(len(X_sample), noise_type)
    Y_sample = target_func(X_sample).reshape(-1, 1) + noise.reshape(-1, 1)
    
    # Initialize GP
    gp_model = CustomGPRegressor(noise_var=1e-2)
    
    for i in range(n_iters):
        # Fit model (automatically optimizes length scale & variance)
        gp_model.fit(X_sample, Y_sample)
        
        # Propose next point via EI
        X_next = propose_location(X_sample, Y_sample, gp_model, bounds)
        
        # Evaluate true function + noise
        noise_next = generate_noise(1, noise_type)
        Y_next = target_func(X_next) + noise_next
        
        # Update samples
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))
        
        best_so_far = np.min(Y_sample)
        print(f"Iter {i+1:2d} | Best observed: {best_so_far:.4f} | Length-Scale: {gp_model.length_scale:.2f}")
    return X_sample, Y_sample

np.random.seed(42)
branin_bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])
run_bo_experiment(branin, branin_bounds, n_iters=20, noise_type='contaminated')
run_bo_experiment(branin, branin_bounds, n_iters=20, noise_type='student_t')
run_bo_experiment(branin, branin_bounds, n_iters=20, noise_type='gaussian')