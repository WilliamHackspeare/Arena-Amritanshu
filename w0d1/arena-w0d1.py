#%%
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from typing import Optional, Callable
import ipywidgets as wg
from fancy_einsum import einsum

import utils
#%%
def DFT_1d(arr : np.ndarray, inverse: bool = False) -> np.ndarray:
    n = arr.size
    exps = -np.outer(np.arange(n),np.arange(n))*(2j*np.pi/n)
    if inverse:
      exps *= -1
    l_matrix = np.exp(exps)
    if inverse:
      l_matrix /= n
    dft = l_matrix@arr
    return dft

def test_DFT_func(DFT_1d):

    y = np.array([1,complex(2,-1),complex(0,-1),complex(-1,2)],dtype=complex)

    y_DFT_actual = DFT_1d(y)
    y_reconstructed_actual = DFT_1d(y_DFT_actual, inverse=True)

    y_DFT_expected = np.array([2,complex(-2,-2),complex(0,-2),complex(4,4)],dtype=complex)
    np.testing.assert_allclose(y_DFT_actual, y_DFT_expected, atol=1e-10, err_msg="DFT failed")
    np.testing.assert_allclose(y_reconstructed_actual, y, atol=1e-10, err_msg="Inverse DFT failed")

test_DFT_func(DFT_1d)

def integrate_function(func: Callable, x0: float, x1: float, n_samples: int = 1000):
    
    x = np.linspace(x0,x1,n_samples,endpoint=False)
    y = func(x)
    step = (x1-x0)/n_samples

    return y.sum()*step

utils.test_integrate_function(integrate_function)

def integrate_product(func1: Callable, func2: Callable, x0: float, x1: float):
    
    return integrate_function(lambda x: func1(x)*func2(x),x0,x1)

utils.test_integrate_product(integrate_product)
# %%
def calculate_fourier_series(func: Callable, max_freq: int = 50):
    a_0 = (1/np.pi) * integrate_function(func, -np.pi, np.pi)
    
    A_n = [(1/np.pi) * integrate_product(func, lambda x: np.cos(n*x), -np.pi, np.pi) for n in range(1, max_freq+1)]
    
    B_n = [(1/np.pi) * integrate_product(func, lambda x: np.sin(n*x), -np.pi, np.pi) for n in range(1, max_freq+1)]
    
    def func_approx(x):
        y = 0.5 * a_0
        y += (np.array(A_n) * [np.cos(n*x) for n in range(1, max_freq+1)]).sum()
        y += (np.array(B_n) * [np.sin(n*x) for n in range(1, max_freq+1)]).sum()
        return y
    func_approx = np.vectorize(func_approx)
    
    return ((a_0, A_n, B_n), func_approx)


step_func = lambda x: 1 * (x > 0)
utils.create_interactive_fourier_graph(calculate_fourier_series, func = step_func)
# %%
NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

x = np.linspace(-np.pi, np.pi, 2000)
y = TARGET_FUNC(x)

x_cos = np.array([np.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = np.array([np.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = np.random.randn()
A_n = np.random.randn(NUM_FREQUENCIES)
B_n = np.random.randn(NUM_FREQUENCIES)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):

    y_pred = 0.5 * a_0 + x_cos.T @ A_n + x_sin.T @ B_n

    loss = np.square(y - y_pred).sum()

    if step % 100 == 0:
        print(f"loss = {loss:.2f}")
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])
        y_pred_list.append(y_pred)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_a_0 = 0.5 * grad_y_pred.sum()
    grad_A_n = x_cos @ grad_y_pred
    grad_B_n = x_sin @ grad_y_pred

    a_0 -= LEARNING_RATE * grad_a_0
    A_n -= LEARNING_RATE * grad_A_n
    B_n -= LEARNING_RATE * grad_B_n

utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
# %%
import torch
import numpy as np
import math
# %%
NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = TARGET_FUNC(x)

x_cos = torch.stack([torch.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = torch.stack([torch.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = torch.randn((), device=device, dtype=dtype)
A_n = torch.randn((NUM_FREQUENCIES), device=device, dtype=dtype)
B_n = torch.randn((NUM_FREQUENCIES), device=device, dtype=dtype)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):

    y_pred = 0.5 * a_0 + x_cos.T @ A_n + x_sin.T @ B_n

    loss = torch.square(y - y_pred).sum()

    if step % 100 == 0:
        print(f"{loss = :.2f}")
        y_pred_list.append(y_pred)
        coeffs_list.append([a_0.item(), A_n.to("cpu").numpy().copy(), B_n.to("cpu").numpy().copy()])

    grad_y_pred = 2.0 * (y_pred - y)
    grad_a_0 = 0.5 * grad_y_pred.sum()
    grad_A_n = x_cos @ grad_y_pred
    grad_B_n = x_sin @ grad_y_pred

    a_0 -= LEARNING_RATE * grad_a_0
    A_n -= LEARNING_RATE * grad_A_n
    B_n -= LEARNING_RATE * grad_B_n
utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
# %%
