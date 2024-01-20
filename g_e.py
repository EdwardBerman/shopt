import numpy as np
import matplotlib.pyplot as plt

def g_magnitude(e_magnitude):
    return 1 + 0.5 * ((1 / e_magnitude**2) - np.sqrt(4 / e_magnitude**2 + 1 / e_magnitude**4))

e_values = np.linspace(0.1, 10, 1000)  # Avoiding 0 to prevent division by zero
g_values = g_magnitude(e_values)
plt.style.use('seaborn-v0_8-whitegrid')
#plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(8, 6))
plt.plot(e_values, g_values, color="blue", linewidth=5)  # Add linewidth
#plt.plot(e_values, g_values, color="blue")
plt.xlabel("Ellipticity Norm ||e||", fontsize=25)
plt.ylabel("Shear Norm ||g||",fontsize=25)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.grid(which='minor', alpha=0.9)
plt.grid(which='major', alpha=0.9)
plt.tick_params(axis='both', which='major', labelsize=24)  # Increase tick label font size

plt.tight_layout()  # To ensure all labels and legends fit properly

plt.savefig("g_vs_e_plot.png", dpi=300)  # Saves with a resolution of 300 dpi
