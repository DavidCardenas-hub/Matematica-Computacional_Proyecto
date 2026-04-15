import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk


class HistogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Expansión y Ecualización de Histogramas")
        self.root.geometry("1100x700")
        self.root.configure(bg="#f2f2f2")

        self.image_gray = None
        self.image_result = None
        self.image_path = None

        title = tk.Label(
            root,
            text="Procesamiento de Imágenes en Escala de Grises",
            font=("Arial", 18, "bold"),
            bg="#f2f2f2"
        )
        title.pack(pady=10)

        btn_frame = tk.Frame(root, bg="#f2f2f2")
        btn_frame.pack(pady=10)

        self.btn_load = tk.Button(
            btn_frame, text="Cargar imagen", width=20, command=self.load_image
        )
        self.btn_load.grid(row=0, column=0, padx=10, pady=5)

        self.btn_expand = tk.Button(
            btn_frame, text="Expansión de histograma", width=20,
            command=self.apply_expansion, state=tk.DISABLED
        )
        self.btn_expand.grid(row=0, column=1, padx=10, pady=5)

        self.btn_equalize = tk.Button(
            btn_frame, text="Ecualización de histograma", width=20,
            command=self.apply_equalization, state=tk.DISABLED
        )
        self.btn_equalize.grid(row=0, column=2, padx=10, pady=5)

        self.btn_show_hist = tk.Button(
            btn_frame, text="Mostrar histogramas", width=20,
            command=self.show_histograms, state=tk.DISABLED
        )
        self.btn_show_hist.grid(row=0, column=3, padx=10, pady=5)

        self.info_label = tk.Label(
            root,
            text="Cargue una imagen en escala de grises para comenzar.",
            font=("Arial", 11),
            bg="#f2f2f2",
            fg="black"
        )
        self.info_label.pack(pady=5)

        img_frame = tk.Frame(root, bg="#f2f2f2")
        img_frame.pack(pady=10, fill="both", expand=True)

        left_frame = tk.Frame(img_frame, bg="#ffffff", bd=2, relief="groove")
        left_frame.pack(side="left", padx=20, pady=10, fill="both", expand=True)

        right_frame = tk.Frame(img_frame, bg="#ffffff", bd=2, relief="groove")
        right_frame.pack(side="right", padx=20, pady=10, fill="both", expand=True)

        tk.Label(
            left_frame, text="Imagen original", font=("Arial", 14, "bold"), bg="#ffffff"
        ).pack(pady=10)
        self.original_panel = tk.Label(left_frame, bg="#ffffff")
        self.original_panel.pack(pady=10)

        tk.Label(
            right_frame, text="Imagen procesada", font=("Arial", 14, "bold"), bg="#ffffff"
        ).pack(pady=10)
        self.result_panel = tk.Label(right_frame, bg="#ffffff")
        self.result_panel.pack(pady=10)

    def is_grayscale(self, image_bgr):
        """
        Verifica si la imagen es realmente en escala de grises.
        Si los tres canales son iguales en todos los píxeles, se considera gris.
        """
        if len(image_bgr.shape) == 2:
            return True

        if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 3:
            b, g, r = cv2.split(image_bgr)
            return np.array_equal(b, g) and np.array_equal(g, r)

        return False

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Seleccione una imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )

        if not file_path:
            return

        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            messagebox.showerror("Error", "No se pudo cargar la imagen.")
            return

        # Validación de escala de grises
        if not self.is_grayscale(image):
            messagebox.showerror(
                "Imagen no válida",
                "La imagen ingresada no está en escala de grises.\n"
                "Por favor, cargue una imagen en blanco y negro o escala de grises."
            )
            self.image_gray = None
            self.image_result = None
            self.image_path = None
            self.clear_panels()
            self.disable_buttons()
            self.info_label.config(text="Se rechazó la imagen porque es a color.", fg="red")
            return

        # Convertir a escala de grises si venía con 3 canales iguales
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.image_gray = image
        self.image_result = image.copy()
        self.image_path = file_path

        self.display_image(self.image_gray, self.original_panel)
        self.display_image(self.image_result, self.result_panel)

        self.enable_buttons()
        self.info_label.config(
            text="Imagen en escala de grises cargada correctamente.",
            fg="green"
        )

    def apply_expansion(self):
        if self.image_gray is None:
            messagebox.showwarning("Aviso", "Primero debe cargar una imagen.")
            return

        img = self.image_gray.astype(np.float32)
        r_min = np.min(img)
        r_max = np.max(img)

        if r_max == r_min:
            messagebox.showwarning(
                "Aviso",
                "No se puede aplicar expansión porque la imagen tiene una sola intensidad."
            )
            return

        expanded = ((img - r_min) * 255.0 / (r_max - r_min)).astype(np.uint8)

        self.image_result = expanded
        self.display_image(self.image_result, self.result_panel)
        self.info_label.config(
            text=f"Expansión aplicada. Intensidad mínima: {int(r_min)}, máxima: {int(r_max)}.",
            fg="blue"
        )

    def apply_equalization(self):
        if self.image_gray is None:
            messagebox.showwarning("Aviso", "Primero debe cargar una imagen.")
            return

        equalized = cv2.equalizeHist(self.image_gray)
        self.image_result = equalized
        self.display_image(self.image_result, self.result_panel)
        self.info_label.config(
            text="Ecualización de histograma aplicada correctamente.",
            fg="blue"
        )

    def show_histograms(self):
        if self.image_gray is None or self.image_result is None:
            messagebox.showwarning("Aviso", "Primero debe cargar y procesar una imagen.")
            return

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(self.image_gray, cmap="gray")
        plt.title("Imagen original")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.hist(self.image_gray.ravel(), bins=256, range=(0, 256))
        plt.title("Histograma original")
        plt.xlabel("Nivel de gris")
        plt.ylabel("Frecuencia")

        plt.subplot(2, 2, 3)
        plt.imshow(self.image_result, cmap="gray")
        plt.title("Imagen procesada")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.hist(self.image_result.ravel(), bins=256, range=(0, 256))
        plt.title("Histograma final")
        plt.xlabel("Nivel de gris")
        plt.ylabel("Frecuencia")

        plt.tight_layout()
        plt.show()

    def display_image(self, image, panel):
        """
        Muestra la imagen dentro de un Label de tkinter.
        """
        max_size = (420, 420)

        pil_img = Image.fromarray(image)
        pil_img.thumbnail(max_size)

        tk_img = ImageTk.PhotoImage(pil_img)
        panel.config(image=tk_img)
        panel.image = tk_img

    def clear_panels(self):
        self.original_panel.config(image="")
        self.original_panel.image = None
        self.result_panel.config(image="")
        self.result_panel.image = None

    def enable_buttons(self):
        self.btn_expand.config(state=tk.NORMAL)
        self.btn_equalize.config(state=tk.NORMAL)
        self.btn_show_hist.config(state=tk.NORMAL)

    def disable_buttons(self):
        self.btn_expand.config(state=tk.DISABLED)
        self.btn_equalize.config(state=tk.DISABLED)
        self.btn_show_hist.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = HistogramApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()