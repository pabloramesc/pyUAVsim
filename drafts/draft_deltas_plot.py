from matplotlib import pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec as gs

# Create a figure with 4 subplots (one for each control)
fig, main_axes = plt.subplots(2, 2, figsize=(12, 6))
axes = []
x0, y0, width, height = main_axes[1, 0].get_position().bounds
for k in range(4):
    bounds = [x0, y0 + k*height/4, width, height/4]
    ax = main_axes[1, 0].inset_axes(bounds)
    axes.append(ax)
# fig.tight_layout()

# Set limits for the x-axis
axes[0].set_xlim(-1.0, +1.0)
axes[1].set_xlim(-1.0, +1.0)
axes[2].set_xlim(-1.0, +1.0)
axes[3].set_xlim(0.0, 1.0) # throttle

# Labels for the bars
labels = ['Aileron', 'Elevator', 'Rudder', 'Throttle']

# Initial values for the bars
values = np.zeros(4)

# Create the horizontal bar chart
bars = []
texts = []
for ax, label, value in zip(axes, labels, values):
    bar, = ax.barh(label, value, color='blue')
    bars.append(bar)
    x_text_position = ax.get_xlim()[1] + 0.1 * np.sum(np.abs(ax.get_xlim()))
    text = ax.text(x_text_position, 0, f"{0.0:.2f}", va='center', ha='right')  # Display the value
    texts.append(text)

fig.tight_layout()

# Function to update the bars in real-time
def update_bars():
    for i in range(100):  # Update 100 times
        # Simulate real-time data by randomizing values
        values[0:3] = np.random.uniform(-1.0, +1.0, 3)
        values[3] = np.random.uniform(0.0, 1.0)

        # Update the data for each bar
        for bar, text, value in zip(bars, texts, values):
            bar.set_width(value)
            text.set_text(f"{value:.2f}")

        # Redraw the plot
        plt.pause(0.1)  # Pause for a short interval to simulate real-time update

# Display the plot and start updating
plt.ion()  # Enable interactive mode
update_bars()
plt.ioff()  # Disable interactive mode
plt.show()
