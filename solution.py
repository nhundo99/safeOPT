"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
import scipy.stats
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
LOOP_DURATION = 10


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        kernel_f = kernels.ConstantKernel(0.5) * kernels.RBF(length_scale=0.5) + kernels.WhiteKernel(0.15,'fixed')
        kernel_v = (kernels.ConstantKernel(4) + kernels.RBF(length_scale=0.5)) * kernels.ConstantKernel(np.sqrt(2)) + kernels.WhiteKernel(0.0001,'fixed')
        self.gp_f = GaussianProcessRegressor(kernel_f, n_restarts_optimizer=1)
        self.gp_v = GaussianProcessRegressor(kernel_v, n_restarts_optimizer=1)
        self.X_data = np.array([])
        self.F_data = np.array([])
        self.V_data = np.array([])
        self.first_added_data = 0.0
        self.latest_save_data = 0.0
        self.first_latest_identical = True
        self.is_first = True
        pass

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        """
        this is not the correct implementation
        later add that we check wether the point is a good recommendation or not.
        if not we should redo i guess

        do something like:
        predict v on recommendation
        then check with mean satisfies condition
        do while loop with a bool satisfies (or do while idk atm)
        """
        # for now easy implementation
        recommendation = self.optimize_acquisition_function()
        x = np.array(recommendation).reshape(-1,1)
        v_mean, v_std = self.gp_v.predict(x, return_std=True)

        is_safe = (v_mean+(2*np.sqrt(v_std*v_std)))<SAFETY_THRESHOLD

        if is_safe:
            return recommendation
        else:
            loop_count = 0
            while not is_safe and loop_count < LOOP_DURATION:
                recommendation = self.optimize_acquisition_function()

                x = np.array(recommendation).reshape(-1,1)
                v_mean, v_std = self.gp_v.predict(x, return_std=True)

                is_safe = (v_mean+(2*np.sqrt(v_std*v_std)))<SAFETY_THRESHOLD

                loop_count += 1
            if is_safe:
                return recommendation
            else:
                # instead of just returning latest save data, return s point that is around latest save data with high probability to satisfy the constraint
                """
                do something like:
                prob of new_x (at least 0.15 far away from initial data 
                """
                return self.latest_save_data

        """
        loop_count = 0
        loop_count_ended = False
        while unsafe and self.unsafe_evals >= 1 and loop_count<LOOP_DURATION:
            # print('in loop')
            recommendation = self.optimize_acquisition_function()
            x = np.array(recommendation).reshape(-1,1)
            v_mean, v_std = self.gp_v.predict(x, return_std=True)

            unsafe = (v_mean+2.5*np.sqrt(v_std*v_std))>=SAFETY_THRESHOLD
            loop_count += 1
            if loop_count == LOOP_DURATION:
                loop_count_ended = True
                unsafe == False
                recommendation = self.last_save_eval

        
        if unsafe:
            self.unsafe_evals += 1
        else:
            self.last_save_eval = recommendation

        
        return recommendation
        """

        raise NotImplementedError

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        mean_f_pred, std_f_pred = self.gp_f.predict(x, return_std=True)
        mean_v_pred, std_v_pred = self.gp_v.predict(x, return_std=True)

        beta = 3
        gamma = 1
        # lam = 0.5

        if (mean_v_pred + std_f_pred) >= SAFETY_THRESHOLD:
            dec = 1
            lam = 1
        else:
            dec = 1
            lam = 0

        ucb = (mean_f_pred + beta*std_f_pred - lam*(mean_v_pred + gamma*std_v_pred))*dec

        return ucb
        # raise NotImplementedError

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        if self.is_first:
            self.first_added_data = x
            self.latest_save_data = x
            self.is_first = False
        elif v < SAFETY_THRESHOLD:
            self.latest_save_data = x

        

        self.X_data = np.append(self.X_data, x)
        
        self.F_data = np.append(self.F_data, f)
        self.V_data = np.append(self.V_data, v)

        x_2d = np.array(self.X_data).reshape(-1,1)

        self.gp_f.fit(x_2d,self.F_data)
        self.gp_v.fit(x_2d,self.V_data)
        # raise NotImplementedError

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        x_vals = np.linspace(DOMAIN[0, 0], DOMAIN[0, 1], 1000)[:,None]

        mean_f_pred = self.gp_f.predict(x_vals)
        mean_v_pred, std_v_pred = self.gp_v.predict(x_vals, return_std=True)

        # to find a safe optimum
        mean_v_pred -= std_v_pred

        possible_sol = np.where(mean_f_pred < SAFETY_THRESHOLD)[0]

        # optimal solution and also in possible solution
        optim_idx = np.argmax(mean_f_pred[possible_sol])
        optim_idx_global = possible_sol[optim_idx]

        sol = x_vals[optim_idx_global].item()
        return sol
        raise NotImplementedError

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
