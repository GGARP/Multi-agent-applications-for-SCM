import numpy as np
import math

# Set the global random seed.
np.random.seed(0)

def gradual_increase(t, D0=5, r=1):
    """Linearly increasing demand over time."""
    return D0 + r * t

def cyclical_demand(t, A=5, T=12, C=5):
    """Sinusoidal (seasonal) demand fluctuations."""
    return int(round(A * math.sin(2 * math.pi * t / T) + C))

def demand_shock(t, D_base=5, S=8, t0=8, delta_t=2):
    """Demand jumps (or drops) for a short interval starting at t0."""
    if t0 <= t <= t0 + delta_t:
        return D_base + S
    else:
        return D_base 
class INARDemandGenerator:
    def __init__(self, alpha=0.5, lam=2, n_periods=13):
        self.alpha = alpha
        self.lam = lam
        self.n_periods = n_periods
        self.generate_new_series()

    def generate_new_series(self):
        """Regenerates demand each time it's called."""
        X = np.zeros(self.n_periods, dtype=int)
        X[0] = np.random.poisson(self.lam)
        for t in range(1, self.n_periods):
            survivors = np.random.binomial(X[t-1], self.alpha)
            new_arrivals = np.random.poisson(self.lam)
            X[t] = survivors + new_arrivals
        self.series = X

    def __call__(self, t):
        """Return demand at t, regenerating per episode."""
        if t == 0:  
            self.generate_new_series()  
        return self.series[t]

env_configs = {
    'constant_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [12, 12, 12, 12],
        'lead_times': [2, 2, 2, 2],
        'demand_fn': lambda t: 4,
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [0, 0, 0, 0],
        'order_costs': [0, 0, 0, 0],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        'comm_size': 3,
    },
    'variable_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [12, 12, 12, 12],
        'lead_times': [2, 2, 2, 2],
        # Use the global RNG here.
        'demand_fn': lambda t: np.random.randint(0, 5),
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [0, 0, 0, 0],
        'order_costs': [0, 0, 0, 0],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        'comm_size': 3,
    },
    'larger_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [20, 20, 20, 20],
        'lead_times': [2, 2, 2, 2],
        # Use the global RNG here as well.
        'demand_fn': lambda t: np.random.randint(0, 9),
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [5, 5, 5, 5],
        'order_costs': [5, 5, 5, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        'comm_size': 3,
    },
    'seasonal_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [12, 12, 12, 12],
        'lead_times': [2, 2, 2, 2],
        # Conditional RNG calls using the global RNG.
        'demand_fn': lambda t: np.random.randint(0, 5) if t <= 4 else np.random.randint(5, 9),
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [5, 5, 5, 5],
        'order_costs': [5, 5, 5, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        'comm_size': 3,
    },
    'normal_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [12, 14, 16, 18],
        'lead_times': [1, 2, 3, 4],
        # Using the global RNG for normal distribution.
        'demand_fn': lambda t: max(0, int(np.random.normal(4, 2))),
        'prod_capacities': [20, 22, 24, 26],
        'sale_prices': [9, 8, 7, 6],
        'order_costs': [8, 7, 6, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        'comm_size': 3,
    },
    'increasing_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [15, 15, 15, 15],
        'lead_times': [2, 2, 2, 2],
        # This function is deterministic.
        'demand_fn': lambda t: gradual_increase(t, D0=5, r=1),
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [9, 8, 7, 6],
        'order_costs': [8, 7, 6, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        'comm_size': 3,
    },
    'cyclical_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [17, 17, 17, 17],
        'lead_times': [2, 2, 2, 2],
        'demand_fn': lambda t: cyclical_demand(t, A=5, T=12, C=5),
        'prod_capacities': [20, 25, 30, 35],
        'sale_prices': [5, 5, 5, 5],
        'order_costs': [5, 5, 5, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        'comm_size': 3,
    },
    'demand_shock': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [12, 12, 12, 12],
        'lead_times': [2, 2, 2, 2],
        'demand_fn': lambda t: demand_shock(t, D_base=5, S=8, t0=8, delta_t=2),
        'prod_capacities': [20, 22, 24, 26],
        'sale_prices': [5, 5, 5, 5],
        'order_costs': [5, 5, 5, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        'comm_size': 3,
    },
    'stochastic_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [20, 20, 20, 20],
        'lead_times': [2, 2, 2, 2],
        # Here we use the pre-generated INARdemand_series.
        'demand_fn': INARDemandGenerator(),
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [5, 5, 5, 5],
        'order_costs': [5, 5, 5, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        'comm_size': 3,
    },
}

# Example: Evaluate one of the demand functions.
# Note: Within a single run, calling the same lambda (e.g., with t=0)
# multiple times will advance the RNG state, so you may get different values.
print("Variable demand for t=0:", env_configs['variable_demand']['demand_fn'](0))

