"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import time
import multiprocessing as mp

from simulator.aircraft import load_airframe_parameters_from_yaml
from simulator.cli import SimConsole

from simulator.plot_process import plot_process
from simulator.simulation_process import simulation_process

if __name__ == "__main__":
    dt = 0.01  # Define el valor de dt
    aerosonde_params = load_airframe_parameters_from_yaml(r"config/aerosonde_parameters.yaml")
    waypoints_file = r"config/go_waypoint.wp"

    plot_queue = mp.Queue()
    plot_proc = mp.Process(target=plot_process, args=(plot_queue,))
    plot_proc.start()

    sim_queue = mp.Queue()
    sim_process = mp.Process(target=simulation_process, args=(sim_queue, dt, aerosonde_params, waypoints_file))
    sim_process.start()

    cli = SimConsole()
    t0 = time.time()

    try:
        while True:
            if not sim_queue.empty():
                state = sim_queue.get()
                plot_queue.put(state)

            t_real = time.time() - t0
            cli.clear()
            cli.print_time(t_real, t_real, dt, 0)  # Actualiza con el tiempo real
            # Aquí puedes agregar más funciones del CLI si es necesario

            time.sleep(0.1)  # Pausa para evitar un bucle demasiado rápido
    finally:
        plot_queue.put(None)
        plot_proc.join()
        sim_process.terminate()
        sim_process.join()