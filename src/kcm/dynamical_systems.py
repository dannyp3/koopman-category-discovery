import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import scipy
from scipy.integrate import solve_ivp
import pickle
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

plt.style.use('dark_background')



def create_one_dimensional_dataset(n,systems,start,end,num_series,noise_multiplier,seed=42):
    rng = np.random.default_rng(seed)


    dataset = {system : [] for system in systems.keys()}
    
    for system, (ode, sampler) in systems.items():

        if system == 'van_der_pol_oscillator':
            t_eval = np.linspace(start,3*end,3*n)
            true_end = 3*end
        else:
            t_eval = np.linspace(start,end,n)
            true_end = end
    
        for _ in tqdm(range(num_series), desc=f'{system}'):

            if system in ['sine_pendulum','sigmoid_oscillator']:
                y0 = [rng.uniform(-1,1), rng.uniform(-1,1)]
            else:
                y0 = [rng.uniform(-5,5), rng.uniform(-5,5)]
            params = sampler(rng)
            
            def wrapped_ode(t,y): return ode(t,y,**params)
        
            sol = solve_ivp(wrapped_ode, [start,true_end], y0, t_eval=t_eval,
                            rtol=1e-8, atol=1e-10, max_step=0.1)
        
            if system == 'van_der_pol_oscillator':
                t = np.linspace(start,end,n)
                y = sol.y[:,-n:]
            else:
                t = sol.t
                y = sol.y

            # if include_noise:
            #     noise = rng.normal(loc=0, scale=0.01, size=y.shape)
            #     y += noise

            signal_std = np.std(y, axis=1, keepdims=True)  # shape: (n_channels, 1)
            noise_scale = noise_multiplier * signal_std # make noise 5% of the max amplitude
            noise = rng.normal(loc=0, scale=noise_scale, size=y.shape)
            y += noise
            
            record = {
                'params' : params,
                'y0' : y0,
                'start' : start,
                'end' : end,
                't' : t,
                'y' : y
            }
        
            dataset[system].append(record)

    return dataset





def create_three_dimensional_dataset_OLD(n,systems,start,end,num_series,noise_multiplier,seed=42,threshold=100):
    rng = np.random.default_rng(seed)


    dataset = {system : [] for system in systems.keys()}
    
    for system, (ode, sampler) in systems.items():

        if system in ['rossler','sprott_a']:
            true_end = 2*end
            t_eval = np.linspace(start,true_end,2*n)
        else:
            t_eval = np.linspace(start,end,n)
            true_end = end

        
        for _ in tqdm(range(num_series), desc=f'{system}'):

            # Make sure the dynamics don't go veyond a particular threshold
            system_norm = threshold * 1.1
            while system_norm > threshold:
                y0 = [rng.uniform(-2,2),
                      rng.uniform(-2,2),
                      rng.uniform(-2,2)]
                
                params = sampler(rng)
                
                def wrapped_ode(t,y): return ode(t,y,**params)

                sol = solve_ivp(wrapped_ode, [start,true_end], y0, t_eval=t_eval,
                                rtol=1e-8, atol=1e-10, max_step=0.05)
                
                if system in ['rossler','sprott_a']:
                    t = np.linspace(start,end,n)
                    y = sol.y[:,::2]
                else:
                    t = sol.t
                    y = sol.y

                system_norm = np.linalg.norm(y, axis=0).max() > threshold
                if system_norm > threshold:
                    print(f'System {system} had norm {system_norm}: regenerating')


        #     noise = rng.normal(loc=0, scale=0.01, size=y.shape)
        #     y += noise

            signal_std = np.std(y, axis=1, keepdims=True)  # shape: (n_channels, 1)
            noise_scale = noise_multiplier * signal_std # make noise 5% of the max amplitude
            noise = rng.normal(loc=0, scale=noise_scale, size=y.shape)
            y += noise
            
            record = {
                'params' : params,
                'y0' : y0,
                'start' : start,
                'end' : end,
                't' : t,
                'y' : y
            }
        
            dataset[system].append(record)

    return dataset


def solve_system(wrapped_ode, start, true_end, y0, t_eval):
    sol = solve_ivp(wrapped_ode, [start,true_end], y0, t_eval=t_eval,
                    rtol=1e-8, atol=1e-10, max_step=0.05)
    return sol
    

def create_three_dimensional_dataset(n,systems,start,end,num_series,noise_multiplier,seed=42,threshold=100):
    rng = np.random.default_rng(seed)


    dataset = {system : [] for system in systems.keys()}
    
    for system, (ode, sampler) in systems.items():

        if system in ['rossler','sprott_a']:
            true_end = 2*end
            t_eval = np.linspace(start,true_end,2*n)
        else:
            t_eval = np.linspace(start,end,n)
            true_end = end
        
        # Make sure the dynamics don't go veyond a particular threshold
        system_norm = threshold * 1.1
        y = None
        # for _ in tqdm(range(num_series), desc=f'{system}'):

        num_regenerated = 0
        with tqdm(total=num_series, desc=f"{system} system") as pbar:
            while True:
    
                y0 = [rng.uniform(-2,2),
                      rng.uniform(-2,2),
                      rng.uniform(-2,2)]
                    
                params = sampler(rng)
                
                def wrapped_ode(t, y): return ode(t, y, **params)

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executer:
                    future = executer.submit(solve_system, wrapped_ode, start, true_end, y0, t_eval)
                    try:
                        sol = future.result(timeout=15)
                    except concurrent.futures.TimeoutError:
                        print('Timeout error -> retrying')
                        continue

                # sol = solve_ivp(wrapped_ode, [start,true_end], y0, t_eval=t_eval,
                #                 rtol=1e-8, atol=1e-10, max_step=0.05)
    
                if not sol.success:
                    print(f'System {system} failed to solve: retrying')
                    continue
                
                if system in ['rossler','sprott_a']:
                    t = np.linspace(start,end,n)
                    y = sol.y[:,::2]
                else:
                    t = sol.t
                    y = sol.y
    
                if y.shape[1] != n:
                    # print(f'System {system} returned {y.shape[1]} samples instead of {n}: retrying')
                    num_regenerated += 1
                    continue
    
                system_norm = np.linalg.norm(y, axis=0).max()
                if system_norm > threshold:
                    # print(f'System {system} had norm {system_norm:.2f}: regenerating')
                    num_regenerated += 1
                    continue
    
                signal_std = np.std(y, axis=1, keepdims=True)  # shape: (n_channels, 1)
                noise_scale = noise_multiplier * signal_std # make noise 5% of the max amplitude
                noise = rng.normal(loc=0, scale=noise_scale, size=y.shape)
                y += noise
                
                record = {
                    'params' : params,
                    'y0' : y0,
                    'start' : start,
                    'end' : end,
                    't' : t,
                    'y' : y
                }
                
                dataset[system].append(record)
                pbar.update(1)
    
                if len(dataset[system]) == num_series:
                    break  # success

            pbar.close()
            # print(f'{num_regenerated} {system} systems regenerated')
        

    return dataset





######## 1-Dimensional Systems (2nd Order) ########

# System Dynamics
def harmonic_oscillator(t, y, omega):
    x, v = y
    return np.array([v, -omega**2 * x])

def sm_damper_forcing(t, y, zeta, omega, gamma, alpha):
    x, v = y
    return [v, -2 * zeta * omega * v - omega**2 * x + gamma * np.cos(alpha * x)]

def duffing_oscillator(t, y, zeta, omega, beta, gamma, alpha):
    x, v = y
    return [v, -2 * zeta * omega * v - omega**2 * x - beta * x**3 + gamma * np.cos(alpha * x)]

def van_der_pol(t, y, mu, z, m):
    x, v = y
    return [v, mu * (1 - m*x**2) * v - z*x]

def nonlinear_damped_oscillator(t, y, zeta, k, gamma, omega):
    x, v = y
    return [v, -zeta*(v / np.sqrt(v**2 + 1e-3)) - k*x + gamma * np.cos(omega * x)]

def nonlinear_spring(t, y, k, alpha, gamma, omega):
    x, v = y
    return [v, -k*x - alpha*x**3 + gamma * np.cos(omega * x)]

def piecewise_linear_oscillator(t, y, k1, k2):
    x, v = y
    acc = -k1*x if x > 0 else -k2*x
    return [v, acc]

def sine_pendulum(t, y, m, zeta):
    theta, theta_v = y
    return [theta_v, -zeta * theta_v -m * np.sin(theta)]

def sigmoid_pendulum(t, y, k, zeta, gamma, omega):
    x, v = y
    return [v, -zeta * v -k * np.tanh(x) + gamma * np.cos(omega * x)]

def arctangent_oscillator(t, y, k, zeta):
    x, v = y
    return [v, -zeta * v - k * np.arctan(x)]


# Time Scaled Omega
def time_scaled_omega(t, omega0, omega1, T):
    return omega0 + (omega1 - omega0) * (t / T)

def time_varying_harmonic(t, y, omega0, omega1, T):
    x, v = y
    omega = time_scaled_omega(t, omega0, omega1, T)
    return np.array([v, -omega**2 * x])

    
# def damped_harmonic_oscillator(t, y, zeta, omega):
#     x, v = y
#     return [v, -2 * zeta *| omega * v - omega**2 * x]

# def double_well_potential(t, y, a, b):
#     x, v = y
#     return [v, -a*x + b*x**3]

# def nonlinear_drag_model(t, y, c, g):
#     x, v = y
#     return [v, c*v*abs(v) + g]



# Parameters
def harmonic_params(rng):
    return {'omega': rng.uniform(2.0, 5.0)}

def under_damped_harmonic_params(rng):
    return {'zeta': rng.uniform(0.0005, 0.001), # (0.02, 0.25),
            'omega': rng.uniform(2.0, 5.0)}

def spring_mass_forcing_params(rng):
    return {'zeta': rng.uniform(0.0005, 0.001), # (0.01, 0.02)
            'omega': rng.uniform(2.0, 5.0),
            'gamma': rng.uniform(0.5, 1.5),
            'alpha': rng.uniform(1, 3)}

def duffing_params(rng):
    return {'zeta' : rng.uniform(0.0005, 0.001), # (0.01, 0.02)
            'omega': rng.uniform(1, 3),
            'beta' : rng.uniform(2, 10),
            'gamma': rng.uniform(0.5, 1.5),
            'alpha': rng.uniform(1, 3)}

def vdp_params(rng):
    return {'mu': rng.uniform(0.5, 2),
            'z' : rng.uniform(4, 20),
            'm' : rng.uniform(4, 20)}

def nonlinear_damped_params(rng):
    return {'zeta': rng.uniform(0.0005, 0.001), # (0.05, 0.15)
            'k' : rng.uniform(4, 7),
            'gamma' : rng.uniform(0.5, 1.5),
            'omega' : rng.uniform(1, 3)}

def nonlinear_spring_params(rng):
    return {'k': rng.uniform(0.5, 2),
            'alpha' : rng.uniform(0.2, 1.0),
            'gamma' : rng.uniform(0.5, 1.5),
            'omega' : rng.uniform(1, 3)}

def piecewise_linear_params(rng):
    return {'k1': rng.uniform(3, 8) ,
            'k2' : rng.uniform(1, 3)}

def sine_pendulum_params(rng):
    return {'m' : rng.uniform(8,12),
            'zeta' : rng.uniform(0.0005, 0.001)} # 0.01, 0.02

def sigmoid_pendulum_params(rng):
    return {'k' : rng.uniform(8,12),
            'zeta' : rng.uniform(0.0005, 0.001), # 0.01, 0.02
            'gamma' : rng.uniform(0.5, 1.5),
            'omega' : rng.uniform(1, 3)}

def arctangent_oscillator_params(rng):
    return {'k': rng.uniform(5, 10),
            'zeta': rng.uniform(0.0005, 0.001)} # 0.01, 0.03

# Varying parameters over time
def time_varying_harmonic_params(rng):
    return {'omega0': rng.uniform(0.1, 0.2),
            'omega1': rng.uniform(6, 8),
            'T' : 50}



# def double_well_params(rng):
#     return {'a': rng.uniform(0.1, 0.4) ,
#             'b' : rng.uniform(0.2, 0.6)}


# def nonlinear_drag_params(rng):
#     return {'c': rng.uniform(0.05, 0.5) ,
#             'g' : rng.uniform(3, 15)} # gravity - should be 9.81 on earth

###################################################







######## 3-Dimensional Systems (1st Order) ########

# System Dynamics
def lorenz(t, y, sigma, rho, beta):
    x, y_, z = y
    dx = sigma * (y_ - x)
    dy = x * (rho - z) - y_
    dz = x * y_ - beta * z
    return np.array([dx, dy, dz])

def rossler(t, y, a, b, c):
    x, y_, z = y
    dx = -y_ - z
    dy = x + a * y_
    dz = b + z * (x - c)
    return [dx, dy, dz]

def chen(t, y, a, b, c):
    x, y_, z = y
    dx = a * (y_ - x)
    dy = (c - a) * x - x * z + c * y_
    dz = x * y_ - b * z
    return [dx, dy, dz]

def chua(t, y, alpha, beta, m1, m2):
    x, y_, z = y
    dx = alpha * (y_ - x - m1 * x + m2 * x**3)
    dy = x - y_ + z
    dz = -beta * y_
    return [dx, dy, dz]

def halvorsen(t, y, a):
    x, y_, z = y
    dx = -a * x - 4 * y_ - 4 * z - y_**2
    dy = -a * y_ - 4 * z - 4 * x - z**2
    dz = -a * z - 4 * x - 4 * y_ - x**2
    return [dx, dy, dz]

def sprott_a(t, y):
    x, y_, z = y
    dx = y_
    dy = -x + y_ * z
    dz = 1 - y_**2
    return [dx, dy, dz]

def tigan(t, y, a, b, c):
    x, y_, z = y
    dx = a * (y_ - x)
    dy = (c - a) * x - x * z + c * y_
    dz = x * y_ - b * z
    return [dx, dy, dz]

def cross_coupled(t, y, a, b, c, gamma, omega):
    x, y_, z = y
    dx = -a * x + np.sin(b * y_) + gamma * np.cos(omega * t)
    dy = -b * y_ + np.sin(c * z)
    dz = -c * z + np.sin(a * x)
    return [dx, dy, dz]


# Parameters
def lorenz_params(rng):
    return {'sigma': rng.uniform(8, 12),
            'rho' : rng.uniform(24, 30),
            'beta' : rng.uniform(2, 3)}

def rossler_params(rng):
    return {'a': rng.uniform(0.1, 0.35),
            'b': rng.uniform(0.1, 0.2), # 0.1, 0.3
            'c': rng.uniform(5.5, 7.5)} # 3.5, 6.0

def chen_params(rng):
    return {'a': rng.uniform(30, 40),
            'b': rng.uniform(1, 4),
            'c': rng.uniform(20, 30)}

def chua_params(rng):
    return {'alpha': rng.uniform(9, 12),
            'beta': rng.uniform(12, 20),
            'm1': rng.uniform(0.5, 1.5),
            'm2': rng.uniform(0.03, 0.08)}

def halvorsen_params(rng):
    return {'a': rng.uniform(1.2, 1.6)}

def sprott_a_params(rng):
    return {}
    
def tigan_params(rng):
    return {'a': rng.uniform(30, 40),
            'b': rng.uniform(1, 5),
            'c': rng.uniform(25, 30)}

def cross_coupled_params(rng):
    return {'a': rng.uniform(1, 3),
            'b': rng.uniform(1, 3),
            'c': rng.uniform(1, 3),
            'gamma': rng.uniform(0.5, 1.5),
            'omega': rng.uniform(1, 3)}

###################################################





######## 3-Dimensional Systems (2nd Order) ########

# System Dynamics
def second_order_chaotic_system(t, y, **params):
    x, y_, z, vx, vy, vz = y
    ax = -k1 * x + alpha * np.sin(y_) - zeta * vx
    ay = -k2 * y_ + alpha * np.sin(z) - zeta * vy
    az = -k3 * z + alpha * np.sin(x) - zeta * vz
    return [vx, vy, vz, ax, ay, az]

# Parameters
def second_order_chaotic_params(rng):
    return {'k1' : rng.uniform(0.5, 2),
            'k2' : rng.uniform(0.5, 2),
            'k3' : rng.uniform(0.5, 2),
            'alpha' : rng.uniform(2, 3),
            'zeta' : rng.uniform(0.01, 0.02)}

###################################################




# Plotting


def plot_1d_trajectories(dataset, n_examples, dim, frac, pos):
    for key in dataset.keys():
    
        curr_data = dataset[key]
        
        plt.figure(figsize=(3,2))
        for i in range(n_examples):
            x = curr_data[i]['t']
            y = curr_data[i]['y'][dim,:].T

            plotting_fraction = int(frac*len(x))

            start = plotting_fraction * pos
            end = plotting_fraction * (pos+1)

            plt.plot(x[start:end],y[start:end],linewidth=1)
    
        plt.title(key)
        # plt.legend(['x','v'])
        plt.show()

    
def plot_phase_diagrams(dataset, n_examples, skip):
    n_examples = 3                          # first N samples of each system
    skip       = 200                         # draw an arrow every `skip` points
    systems_enum    = list(dataset.keys())
    rows, cols = len(systems_enum), n_examples
    fig, axes  = plt.subplots(rows, cols,
                              figsize=(4*cols, 3*rows),
                              sharex='col', sharey='col')
    fig.suptitle(f'Phase diagrams (with arrows) – first {n_examples} samples', fontsize=14)
    
    for r, sys in enumerate(systems_enum):
        records = dataset[sys][:n_examples]
    
        for c, rec in enumerate(records):
            x, v = rec['y']                        # position & velocity
            ax   = axes[r, c] if rows > 1 else axes[c]
    
            # main trajectory line
            ax.plot(x, v, lw=1, color='white')
    
            # add arrows along the trajectory
            for j in range(0, len(x) - 1, skip):
                ax.annotate("",
                            xy=(x[j + 1], v[j + 1]),
                            xytext=(x[j], v[j]),
                            arrowprops=dict(arrowstyle="->",
                                            color="cyan",
                                            lw=0.8))
    
            # labels / titles
            if c == 0:
                ax.set_ylabel(sys.replace('_', ' '), fontsize=10)
            if r == rows - 1:
                ax.set_xlabel('x')
            if r == 0:
                ax.set_title(f'Sample {c + 1}')
            ax.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




def plot_3d_trajectories(dataset, n_examples, frac, pos):
    
    for key in dataset.keys():
    
        curr_data = dataset[key]
        
        fig, ax = plt.subplots(n_examples,1,figsize=(16,n_examples*1.3))
        for i in range(n_examples):
            t = curr_data[i]['t']
            x, y, z = curr_data[i]['y']

            plotting_fraction = int(frac*len(x))

            start = plotting_fraction * pos
            end = plotting_fraction * (pos+1)
            
            ax[i].plot(t[start:end],x[start:end],linewidth=1)
            ax[i].plot(t[start:end],y[start:end],linewidth=1)
            ax[i].plot(t[start:end],z[start:end],linewidth=1)
            ax[i].legend(['x','y','z'])
    
        plt.suptitle(f'{key.capitalize()} System')
        plt.show()

def plot_3d_time_colored_trajectories(dataset, n_examples=3, cmap=cm.plasma):
    """
    Plots 3D phase trajectories for each system in the dataset.
    Each trajectory is color-coded by time.
    
    Parameters:
        dataset     (dict): {system_name: list of records, each with 'y' and 't'}
        n_examples   (int): Number of trajectories per system to plot
        cmap         (Colormap): Matplotlib colormap for time
    """
    systems_enum = list(dataset.keys())
    rows, cols = len(systems_enum), n_examples
    fig = plt.figure(figsize=(5 * cols, 4 * rows))
    fig.suptitle("3D Phase Trajectories (colored by time)", fontsize=14)

    for r, sys in enumerate(systems_enum):
        records = dataset[sys][:n_examples]

        for c, rec in enumerate(records):
            x, y, z = rec['y']  # shape (3, N)
            t = rec['t']
            idx = r * cols + c + 1
            ax = fig.add_subplot(rows, cols, idx, projection='3d')

            # Color trajectory segments by time
            norm_t = (t - t[0]) / (t[-1] - t[0])
            for i in range(len(t) - 1):
                ax.plot(x[i:i+2], y[i:i+2], z[i:i+2],
                        color=cmap(norm_t[i]), linewidth=0.7)

            ax.set_title(f'{sys.replace("_", " ")}\nSample {c + 1}', fontsize=9)
            ax.set_xlabel("x", fontsize=8)
            ax.set_ylabel("y", fontsize=8)
            ax.set_zlabel("z", fontsize=8)
            ax.tick_params(labelsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
