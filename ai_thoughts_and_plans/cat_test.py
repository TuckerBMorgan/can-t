import numpy as np
import matplotlib.pyplot as plt

def swiglu(x):
    return x * (x / (1 + np.exp(-x)))

x = np.linspace(-6, 6, 500)
y = swiglu(x)

plt.plot(x, y)
plt.title("SwiGLU Activation Function")
plt.xlabel("x")
plt.ylabel("f(x) = x * SiLU(x)")
plt.grid(True)
plt.show()
