import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import G, c, hbar, m_e, m_u

MU_E = 2          # mean molecular weight per electron (He/C/O)
X_SURF = 1e-3     # stop when x = p_F/(m_e c) falls below this ("surface")

# Chandrasekhar (exact free degenerate electron gas) constants
A = m_e**4 * c**5 / (24 * np.pi**2 * hbar**3)


def delete_last_line():
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')


def P_of_x(x):
    """Exact electron degeneracy pressure. Uses a series for small x because
    the closed form cancels catastrophically there (P ~ x^5)."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    series = A * (8/5*x**5 - 4/7*x**7 + x**9/3 - 5/22*x**11)
    exact = A * (x*(2*x**2 - 3)*np.sqrt(1 + x**2) + 3*np.arcsinh(x))
    out = np.where(x < 0.05, series, exact)
    return out if out.ndim else float(out)


def dPdx(x):
    return 8*A*x**4 / np.sqrt(1 + x**2)


def eps_of_x(x, mu_e=MU_E):
    """Energy density = rest-mass density of the ions * c^2."""
    rho = mu_e * m_u * (m_e*c/hbar)**3 * x**3 / (3*np.pi**2)
    return rho * c**2


def x_of_eps(eps, mu_e=MU_E):
    """Inverse of eps_of_x (so find_eps_c can keep passing eps_c)."""
    return (3*np.pi**2*eps/(mu_e*m_u*c**2))**(1/3) * hbar/(m_e*c)


def solve(eps_c, allpoints=False, mu_e=MU_E):
    # `iter` is kept only so existing calls don't break; it is not used.
    delete_last_line()
    print(f"Central value of Epsilon {'Final' if allpoints else 'Guess'} = {eps_c}")

    def f(r, y):
        M, x = y
        x = max(x, 1e-12)
        P, eps = P_of_x(x), eps_of_x(x, mu_e)
        dm = 4*np.pi*r**2*eps/c**2
        dPdr = -(G/(c**2*r**2))*(eps + P)*(M + 4*np.pi*r**3*P/c**2) \
               / (1 - 2*G*M/(r*c**2))
        return [dm, dPdr/dPdx(x)]

    def surface(r, y):
        return y[1] - X_SURF
    surface.terminal = True
    surface.direction = -1

    x_c = x_of_eps(eps_c, mu_e)
    r0 = 1.0
    m0 = 4*np.pi/3*r0**3*eps_c/c**2

    sol = solve_ivp(f, (r0, 1e11), [m0, x_c], events=surface,
                    method='LSODA', rtol=1e-8,
                    atol=[1e-10*1.989e30, 1e-10],
                    dense_output=allpoints)

    if len(sol.t_events[0]) == 0:
        return (np.nan, np.nan, None, None) if allpoints else (np.nan, np.nan)

    R = sol.t_events[0][0]
    M = sol.y_events[0][0][0]

    if allpoints:
        ys = np.array([sol.y[0], P_of_x(sol.y[1])])   # [mass, pressure] as before
        return (R, M, sol.t, ys)
    return (R, M)