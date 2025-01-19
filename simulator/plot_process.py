import multiprocessing as mp
from matplotlib import pyplot as plt
from simulator.gui.attitude_position_panel import AttitudePositionPanel

def plot_process(queue: mp.Queue):
    gui = AttitudePositionPanel(use_blit=False, pos_3d=True)
    
    plt.ion()  # Modo interactivo activado
    
    while True:
        data = queue.get()
        if data is None:
            break
        state = data
        gui.add_data(state)
        gui.update(state, pause=0.01)  # Actualiza la GUI con una pausa para refrescar

    plt.ioff()  # Modo interactivo desactivado
    plt.show()  # Mostrar el gráfico cuando termine