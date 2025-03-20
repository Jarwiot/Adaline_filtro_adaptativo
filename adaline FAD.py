import numpy as np
import matplotlib.pyplot as plt
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Clase ADALINE
class Adaline:
    def __init__(self, lr=0.01, epochs=10):  # Aumentar las épocas
        self.lr = lr
        self.epochs = epochs
        self.weights = np.random.randn(2)  # Ahora tenemos dos pesos (W1 y W2)
        self.bias = np.random.randn()

    def train(self, X, y, ax, canvas, w1_label, w2_label, bias_label):
        errors = []
        for _ in range(self.epochs):
            total_error = 0
            for xi, target in zip(X, y):
                output = self.predict(xi)
                error = target - output
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                total_error += error**2
            errors.append(total_error)

        # Actualizar etiquetas de los valores de W1, W2 y Bias
        w1_label.configure(text=f"{float(self.weights[0]):.2f}")
        w2_label.configure(text=f"{float(self.weights[1]):.2f}")
        bias_label.configure(text=f"{float(self.bias):.4f}")

        # Actualizar la gráfica
        ax.clear()
        ax.plot(y, label='Señal Original', color='blue')
        ax.plot(self.predict(X).flatten(), label='Señal Filtrada', color='red')
        ax.set_title("Corrección de Ruido")
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

X1 = noisy_signal
X2 = np.roll(noisy_signal, 1)  # Segunda entrada: una versión desplazada de X1
X = np.column_stack((X1, X2))  # Características con dos entradas

y = signal.reshape(-1, 1)  # Convertir y en una matriz columna

# Interfaz gráfica
ctk.set_appearance_mode("dark")
root = ctk.CTk()
root.title("ADALINE FILTRO ADAPTATIVO")
root.geometry("750x600")

frame = ctk.CTkFrame(root)
frame.pack(pady=10, fill='both', expand=True)

# Etiqueta del título
title_label = ctk.CTkLabel(frame, text="ADALINE FILTRO ADAPTATIVO", font=("Arial", 20, "bold"))
title_label.pack(pady=5)

# Crear un frame para mostrar valores
values_frame = ctk.CTkFrame(frame)
values_frame.pack(pady=10)

w1_text = ctk.CTkLabel(values_frame, text="W1", font=("Arial", 14))
w1_text.grid(row=0, column=0, padx=5)
w1_label = ctk.CTkLabel(values_frame, text="0.00", font=("Arial", 14))
w1_label.grid(row=0, column=1, padx=5)

w2_text = ctk.CTkLabel(values_frame, text="W2", font=("Arial", 14))
w2_text.grid(row=1, column=0, padx=5)
w2_label = ctk.CTkLabel(values_frame, text="0.00", font=("Arial", 14))
w2_label.grid(row=1, column=1, padx=5)

bias_text = ctk.CTkLabel(values_frame, text="BIAS", font=("Arial", 14))
bias_text.grid(row=2, column=0, padx=5)
bias_label = ctk.CTkLabel(values_frame, text="0.0000", font=("Arial", 14))
bias_label.grid(row=2, column=1, padx=5)

# Gráfico inicial
fig, ax = plt.subplots()
ax.plot(noisy_signal, label='Señal con Ruido', color='magenta')
ax.set_title("Señal Original con Ruido")
ax.legend()
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().pack()

adaline = Adaline(lr=0.001, epochs=10)  # Más iteraciones
def iniciar_proceso():
    adaline.train(X, y, ax, canvas, w1_label, w2_label, bias_label)

# Botón para iniciar el proceso
boton = ctk.CTkButton(frame, text="PLOT", command=iniciar_proceso)
boton.pack(pady=10)

root.mainloop()
