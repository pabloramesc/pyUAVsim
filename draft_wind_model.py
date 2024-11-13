import numpy as np
from simulator.math.transfer_function import TransferFunction
from simulator.math.lti_systems import zpk2tf


def generate_dryden_gusts(
    t: np.ndarray,
    Va: float,
    h: float,
    turbulence: str = "light",
) -> np.ndarray:
    """
    Generate wind turbulence using the simplified Dryden gust model.

    Simulates wind gusts in the u, v, and w directions based on airspeed, altitude, and turbulence level.

    Parameters
    ----------
    t : np.ndarray
        Time array (seconds) for the simulation.
    Va : float
        Airspeed (m/s).
    h : float
        Altitude (meters).
    turbulence : str, optional
        The turbulence level ("light", "moderate", "severe"). Default is "light".

    Returns
    -------
    np.ndarray
        Array of simulated wind gusts in the u, v, and w directions (m/s) for each time step.

    Raises
    ------
    ValueError
        If an invalid turbulence level is provided.
    """

    # Calculate sigma factor
    factors = {"light": 1.0, "moderate": 2.0, "severe": 4.0}
    if turbulence not in factors:
        raise ValueError(f"Invalid turbulence level '{turbulence}'.")
    sigma_factor = factors[turbulence]

    # Base model parameters for interpolation
    base_altitudes = [50.0, 600.0]
    base_Lu = [200.0, 533.0]
    base_Lv = [200.0, 533.0]
    base_Lw = [50.0, 533.0]
    base_sigma_u = [1.06 * sigma_factor, 1.50 * sigma_factor]
    base_sigma_v = [1.06 * sigma_factor, 1.50 * sigma_factor]
    base_sigma_w = [0.70 * sigma_factor, 1.50 * sigma_factor]

    # Interpolated parameters
    Lu = np.interp(h, base_altitudes, base_Lu)
    Lv = np.interp(h, base_altitudes, base_Lv)
    Lw = np.interp(h, base_altitudes, base_Lw)
    sigma_u = np.interp(h, base_altitudes, base_sigma_u)
    sigma_v = np.interp(h, base_altitudes, base_sigma_v)
    sigma_w = np.interp(h, base_altitudes, base_sigma_w)

    # Create transfer functions
    Ku = sigma_u * np.sqrt(2 * Va / Lu)
    Kv = sigma_v * np.sqrt(3 * Va / Lv)
    Kw = sigma_w * np.sqrt(3 * Va / Lw)
    Hu = TransferFunction(*zpk2tf([], [-Va / Lu], Ku))
    Hv = TransferFunction(*zpk2tf([-Va / Lv / np.sqrt(3)], [-Va / Lv] * 2, Kv))
    Hw = TransferFunction(*zpk2tf([-Va / Lw / np.sqrt(3)], [-Va / Lw] * 2, Kw))

    # Generate random noise (representing gust disturbances) for each axis
    noise_u = np.random.normal(0.0, 1.0, t.size)
    noise_v = np.random.normal(0.0, 1.0, t.size)
    noise_w = np.random.normal(0.0, 1.0, t.size)

    # Simulate the gust response, which will have feedback and dynamics
    u_wg = Hu.simulate(noise_u, t)[1]
    v_wg = Hv.simulate(noise_v, t)[1]
    w_wg = Hw.simulate(noise_w, t)[1]

    # Combine the gusts into a single array
    return np.column_stack([u_wg, v_wg, w_wg])


def plot_gusts(gusts: np.ndarray, t: np.ndarray):
    """Plot wind gust components over time."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("Dryden Wind Gust Model")
    labels = [r"$u_{wg}$ (m/s)", r"$v_{wg}$ (m/s)", r"$w_{wg}$ (m/s)"]
    for i, ax in enumerate(axs):
        ax.plot(t, gusts[:, i], label=labels[i])
        ax.set_ylabel(labels[i])
        ax.grid()
    axs[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    plt.show()


# Example
t = np.linspace(0.0, 60.0, 1000)
gusts = generate_dryden_gusts(t, Va=25.0, h=1e3, turbulence="severe")
plot_gusts(gusts, t)
