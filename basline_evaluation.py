import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")



# Provides a multi-seed evaluation method that calculates surrogate model accuracy metrics (RMSE, NLPD, coverage) and optimization performance (simple regret),
# concluding with a plot of average regret trajectories to compare noise robustness across multiple runs.


def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
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
        self.y_train = (y_raw - self.y_mean) / self.y_std
        
        def objective(params):
            ls, var = params
            K = rbf_kernel(self.X_train, self.X_train, ls, var) + self.noise_var * np.eye(len(self.X_train))
            try:
                K_inv = np.linalg.inv(K)
                # The LML is now calculated on the normalized y_train!
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


# 2. Benchmarks, Noise, and Metrics

def branin(X):
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    return (x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5 / np.pi * x1 - 6.0)**2 + 10.0 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10.0

def six_hump_camel(X):
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    return (4 - 2.1 * x1**2 + (x1**4) / 3.0) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2

def generate_noise(n, noise_type='gaussian', scale=0.5, seed=None):
    if seed is not None: np.random.seed(seed)
    if noise_type == 'gaussian':
        return np.random.normal(0.0, scale, n)
    elif noise_type == 'student_t':
        return np.random.standard_t(df=2, size=n) * scale
    elif noise_type == 'contaminated':
        base = np.random.normal(0.0, scale, n)
        outliers = np.random.normal(0.0, scale * 20.0, n)
        return np.where(np.random.rand(n) < 0.10, outliers, base)

def compute_surrogate_metrics(gp_model, X_test, y_test_true):
    """Calculates RMSE, NLPD, and Coverage."""
    mu, std = gp_model.predict(X_test, return_std=True)
    
    rmse = np.sqrt(np.mean((mu - y_test_true)**2))
    nlpd = np.mean(0.5 * np.log(2 * np.pi * std**2) + ((y_test_true - mu)**2) / (2 * std**2))
    
    lower, upper = mu - 1.96 * std, mu + 1.96 * std
    coverage = np.mean((y_test_true >= lower) & (y_test_true <= upper))
    
    return rmse, nlpd, coverage


# 3. Optimization and Multi-Seed Loop

def expected_improvement(X, Y_sample, gp_model, xi=0.01):
    mu, std = gp_model.predict(X, return_std=True)
    imp = np.min(Y_sample) - mu - xi
    Z = imp / std
    ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
    ei[std <= 1e-9] = 0.0
    return ei

def propose_location(X_sample, Y_sample, gp_model, bounds):
    best_val, best_x = -np.inf, None
    def min_obj(x): return -expected_improvement(x.reshape(1, -1), Y_sample, gp_model)[0]
    
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(10, bounds.shape[0])):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if -res.fun > best_val:
            best_val, best_x = -res.fun, res.x
    return best_x.reshape(1, bounds.shape[0])

def run_multiseed_evaluation(target_func, bounds, f_opt, noise_type, n_seeds=5, n_iters=10):
    """Runs the BO experiment across multiple seeds and aggregates the metrics."""
    # Create a dense grid to test the surrogate model's accuracy
    grid_x1 = np.linspace(bounds[0, 0], bounds[0, 1], 30)
    grid_x2 = np.linspace(bounds[1, 0], bounds[1, 1], 30)
    X1, X2 = np.meshgrid(grid_x1, grid_x2)
    X_test = np.column_stack([X1.ravel(), X2.ravel()])
    y_test_true = target_func(X_test)
    
    metrics = {'rmse': [], 'nlpd': [], 'coverage': [], 'regret': []}
    
    for seed in range(42, 42 + n_seeds):
        np.random.seed(seed)
        
        # Initialization
        X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(5, bounds.shape[0]))
        y_true_init = target_func(X_sample)
        noise = generate_noise(len(X_sample), noise_type, seed=seed)
        Y_sample = y_true_init + noise
        
        best_true_vals = [np.min(y_true_init)]
        gp_model = CustomGPRegressor(noise_var=1e-2)
        
        # BO Loop
        for _ in range(n_iters):
            gp_model.fit(X_sample, Y_sample)
            X_next = propose_location(X_sample, Y_sample, gp_model, bounds)
            
            y_next_true = target_func(X_next)[0]
            y_next_noisy = y_next_true + generate_noise(1, noise_type)[0]
            
            X_sample = np.vstack((X_sample, X_next))
            Y_sample = np.append(Y_sample, y_next_noisy)
            
            best_true_vals.append(min(best_true_vals[-1], y_next_true))
            
        # Final Model Fitting for Surrogate Metrics
        gp_model.fit(X_sample, Y_sample)
        rmse, nlpd, cov = compute_surrogate_metrics(gp_model, X_test, y_test_true)
        
        metrics['rmse'].append(rmse)
        metrics['nlpd'].append(nlpd)
        metrics['coverage'].append(cov)
        metrics['regret'].append(best_true_vals[-1] - f_opt)
        
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}



def track_optimization_trajectory(target_func, bounds, f_opt, noise_type, n_iters=20, n_seeds=5):
    all_regrets = np.zeros((n_seeds, n_iters))
    
    for seed in range(n_seeds):
        np.random.seed(seed + 42)
        X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(5, bounds.shape[0]))
        Y_sample = target_func(X_sample) + generate_noise(5, noise_type, seed=seed)
        
        best_true_vals = []
        current_best = np.min(target_func(X_sample))
        
        gp_model = CustomGPRegressor(noise_var=1e-2)
        
        for i in range(n_iters):
            gp_model.fit(X_sample, Y_sample)
            X_next = propose_location(X_sample, Y_sample, gp_model, bounds)
            
            y_next_true = target_func(X_next)[0]
            y_next_noisy = y_next_true + generate_noise(1, noise_type)[0]
            
            X_sample = np.vstack((X_sample, X_next))
            Y_sample = np.append(Y_sample, y_next_noisy)
            
            current_best = min(current_best, y_next_true)
            best_true_vals.append(current_best - f_opt) # Track Regret
            
        all_regrets[seed] = best_true_vals
        
    return np.mean(all_regrets, axis=0), np.std(all_regrets, axis=0)


def run_multiseed_evaluation(target_func, bounds, f_opt, noise_type, n_seeds=5, n_iters=10):
    grid_x1 = np.linspace(bounds[0, 0], bounds[0, 1], 30)
    grid_x2 = np.linspace(bounds[1, 0], bounds[1, 1], 30)
    X1, X2 = np.meshgrid(grid_x1, grid_x2)
    X_test = np.column_stack([X1.ravel(), X2.ravel()])
    y_test_true = target_func(X_test)
    
    metrics = {'rmse': [], 'nlpd': [], 'coverage': [], 'regret': []}
    
    for seed in range(42, 42 + n_seeds):
        np.random.seed(seed)
        
        X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(5, bounds.shape[0]))
        y_true_init = target_func(X_sample)
        noise = generate_noise(len(X_sample), noise_type, seed=seed)
        Y_sample = y_true_init + noise
        
        best_true_vals = [np.min(y_true_init)]
        gp_model = CustomGPRegressor(noise_var=1e-2)
        
        for _ in range(n_iters):
            gp_model.fit(X_sample, Y_sample)
            X_next = propose_location(X_sample, Y_sample, gp_model, bounds)
            
            y_next_true = target_func(X_next)[0]
            y_next_noisy = y_next_true + generate_noise(1, noise_type)[0]
            
            X_sample = np.vstack((X_sample, X_next))
            Y_sample = np.append(Y_sample, y_next_noisy)
            best_true_vals.append(min(best_true_vals[-1], y_next_true))
            
        gp_model.fit(X_sample, Y_sample)
        
        
        rmse, nlpd, cov = compute_surrogate_metrics(gp_model, X_test, y_test_true)
        
        metrics['rmse'].append(rmse)
        metrics['nlpd'].append(nlpd)
        metrics['coverage'].append(cov)
        metrics['regret'].append(best_true_vals[-1] - f_opt)
        
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}


# 4. Execution & Report Generation


print("  Evaluation: GP Baseline         \n")


benchmarks = [
    ('Branin', branin, np.array([[-5.0, 10.0], [0.0, 15.0]]), 0.397887),
    ('Six-Hump Camel', six_hump_camel, np.array([[-3.0, 3.0], [-2.0, 2.0]]), -1.0316)
]
noise_types = ['gaussian', 'student_t', 'contaminated']

for b_name, b_func, b_bounds, b_opt in benchmarks:
    print(f"--- {b_name} Function ---")
    print(f"{'Noise':<15} | {'RMSE':<18} | {'NLPD':<18} | {'Coverage (95%)':<18} | {'Simple Regret':<18}")
    print("-" * 95)
    
    for noise in noise_types:
        res = run_multiseed_evaluation(b_func, b_bounds, b_opt, noise, n_seeds=5, n_iters=10)
        rmse = f"{res['rmse'][0]:.2f} ± {res['rmse'][1]:.2f}"
        nlpd = f"{res['nlpd'][0]:.2f} ± {res['nlpd'][1]:.2f}"
        cov = f"{res['coverage'][0]*100:.1f}% ± {res['coverage'][1]*100:.1f}%"
        reg = f"{res['regret'][0]:.2f} ± {res['regret'][1]:.2f}"
        
        print(f"{noise:<15} | {rmse:<18} | {nlpd:<18} | {cov:<18} | {reg:<18}")
    print("\n")

# bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])
# f_opt = 0.397887

# print("Simulating Trajectories...")
# mean_g, std_g = track_optimization_trajectory(branin, bounds, f_opt, 'gaussian')
# mean_c, std_c = track_optimization_trajectory(branin, bounds, f_opt, 'contaminated')

# # Plotting the results
# plt.figure(figsize=(10, 6))
# iters = np.arange(1, 21)

# plt.plot(iters, mean_g, 'b-', label='Gaussian Noise (Baseline)', linewidth=2)
# plt.fill_between(iters, mean_g - std_g, mean_g + std_g, color='blue', alpha=0.1)

# plt.plot(iters, mean_c, 'r--', label='Contaminated Noise (Outliers)', linewidth=2)
# plt.fill_between(iters, mean_c - std_c, mean_c + std_c, color='red', alpha=0.1)

# plt.yscale('log')
# plt.title('Bayesian Optimization Regret Trajectory (Branin)')
# plt.xlabel('BO Iterations')
# plt.ylabel('Simple Regret (Log Scale)')
# plt.legend()
# plt.grid(True, which="both", ls="--", alpha=0.5)
# plt.show()