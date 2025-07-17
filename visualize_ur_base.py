import numpy as np
import matplotlib.pyplot as plt
from tcp_path_loader import load_tcp_path

# Offset von UR zur MiR-Plattform (in m)
OFFSET_X = 0.3
OFFSET_Y = -0.2

def compute_ur_base(x_mir, y_mir, theta):
    x_ur = x_mir + np.cos(theta) * OFFSET_X - np.sin(theta) * OFFSET_Y
    y_ur = y_mir + np.sin(theta) * OFFSET_X + np.cos(theta) * OFFSET_Y
    return x_ur, y_ur

def generate_dummy_mir_path(tcp_path):
    # einfache Gerade entlang TCP-Pfad (gleiche Richtung, UR bleibt 80 cm weg)
    mir_path = []
    for (x_tcp, y_tcp) in tcp_path:
        # Rückversetze den UR-Basispunkt auf 0.8 m Abstand nach hinten
        dx = 0.8
        theta = np.arctan2(y_tcp - y_tcp, x_tcp - x_tcp + 1e-6)  # konstante Ausrichtung
        x_mir = x_tcp - np.cos(theta) * dx
        y_mir = y_tcp - np.sin(theta) * dx
        mir_path.append((x_mir, y_mir, theta))
    return mir_path

def generate_offset_mir_path(tcp_path, offset_distance=0.8):
    mir_path = []
    for i in range(len(tcp_path)):
        if i < len(tcp_path) - 1:
            dx = tcp_path[i+1][0] - tcp_path[i][0]
            dy = tcp_path[i+1][1] - tcp_path[i][1]
        else:
            dx = tcp_path[i][0] - tcp_path[i-1][0]
            dy = tcp_path[i][1] - tcp_path[i-1][1]
        
        theta = np.arctan2(dy, dx)  # Tangentenrichtung
        normal_angle = theta + np.pi / 2  # Rechte Normalenrichtung
        
        # Plattform liegt rechts vorne → wir versetzen nach rechts
        x_mir = tcp_path[i][0] + np.cos(normal_angle) * offset_distance
        y_mir = tcp_path[i][1] + np.sin(normal_angle) * offset_distance
        mir_path.append((x_mir, y_mir, theta))  # Orientierung entlang Tangente

    return mir_path


def main():
    tcp_path = load_tcp_path()
    mir_path = generate_offset_mir_path(tcp_path)

    ur_base_points = [compute_ur_base(x, y, theta) for (x, y, theta) in mir_path]

    tcp_x, tcp_y = zip(*tcp_path)
    ur_x, ur_y = zip(*ur_base_points)

    plt.figure()
    plt.plot(tcp_x, tcp_y, 'b.-', label="TCP Path")
    plt.plot(ur_x, ur_y, 'r.-', label="UR Base (mounted on MiR)")
    plt.axis('equal')
    plt.legend()
    plt.title("TCP Path vs UR Base Path")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
