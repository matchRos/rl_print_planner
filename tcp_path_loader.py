# tcp_path_loader.py
from xTCP import xTCP
from yTCP import yTCP

def load_tcp_path():
    x = xTCP()
    y = yTCP()
    assert len(x) == len(y), "x and y must have same length"
    tcp_path = list(zip(x, y))
    return tcp_path

if __name__ == "__main__":
    path = load_tcp_path()
    print(f"Loaded {len(path)} TCP points")
    for i, (x, y) in enumerate(path[:5]):
        print(f"  Point {i}: ({x:.3f}, {y:.3f})")
