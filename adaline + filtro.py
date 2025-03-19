import numpy as np
import matplotlib.pyplot as plt
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Parámetros del ADALINE
class Adaline:
    def __init__(self, lr=0.01, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def train(self, X, y, ax, canvas):
        self.weights = np.random.randn(X.shape[1])
        self.bias = 0
        errors = []

        for epoch in range(self.epochs):
            total_error = 0
            for xi, target in zip(X, y):
                output = self.predict(xi)
                error = target - output
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                total_error += error**2
            errors.append(total_error)
            
            # Actualizar la gráfica en tiempo real
            ax.clear()
            ax.plot(y, label='Señal Original', color='blue')
            ax.plot(self.predict(X), label='Señal Filtrada', color='red')
            ax.set_title(f"Corrección de Ruido - epocas {epoch+1}")
            ax.legend()
            canvas.draw()
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Generar datos sintéticos con ruido blanco
np.random.seed(42)
time = np.linspace(0, 10, 500)
signal = np.sin(time)  # Señal original
noise = np.random.normal(0, 0.5, signal.shape)  # Ruido blanco
noisy_signal = signal + noise

# Convertir los datos en una forma adecuada para ADALINE
X = np.array([noisy_signal[:-1]]).T  # Características
y = signal[1:]  # Objetivo

# Crear la interfaz gráfica
ctk.set_appearance_mode("dark")
root = ctk.CTk()
root.title("Filtro Adaptativo ADALINE")
root.geometry("600x500")

frame = ctk.CTkFrame(root)
frame.pack(pady=20, fill='both', expand=True)

fig, ax = plt.subplots()
ax.plot(noisy_signal, label='Señal con Ruido', color='magenta')
ax.set_title("Señal Original con Ruido")
ax.legend()
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().pack()

adaline = Adaline(lr=0.001, epochs=200)
def iniciar_proceso():
    adaline.train(X, y, ax, canvas)

boton = ctk.CTkButton(root, text="Iniciar Corrección", command=iniciar_proceso)
boton.pack(pady=10)

root.mainloop()
