import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk


class HistogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Proyecto de Matemática Computacional")
        self.root.geometry("1250x780")
        self.root.configure(bg="#eef2f7")
        self.root.minsize(1100, 700)

        # Variables principales
        self.image_gray = None
        self.image_result = None
        self.image_path = None
        self.image_bits = None
        self.total_pixels = None

        # Estilo formal con ttk
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.style.configure(
            "Title.TLabel",
            font=("Segoe UI", 20, "bold"),
            background="#eef2f7",
            foreground="#1f2937"
        )

        self.style.configure(
            "Subtitle.TLabel",
            font=("Segoe UI", 11),
            background="#eef2f7",
            foreground="#374151"
        )

        self.style.configure(
            "Card.TFrame",
            background="#ffffff",
            relief="solid",
            borderwidth=1
        )

        self.style.configure(
            "CardTitle.TLabel",
            font=("Segoe UI", 13, "bold"),
            background="#ffffff",
            foreground="#1f2937"
        )

        self.style.configure(
            "Info.TLabel",
            font=("Segoe UI", 10),
            background="#eef2f7",
            foreground="#111827"
        )

        self.style.configure(
            "Formal.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=8
        )

        self.build_interface()

    def build_interface(self):
        # Encabezado
        header_frame = ttk.Frame(self.root, style="Card.TFrame")
        header_frame.pack(fill="x", padx=20, pady=(15, 8))

        title = ttk.Label(
            header_frame,
            text="Expansión y Ecualización de Histogramas en Imágenes en Escala de Grises",
            style="Title.TLabel"
        )
        title.pack(pady=(12, 2))

        subtitle = ttk.Label(
            header_frame,
            text="Curso: Matemática Computacional | Procesamiento digital de imágenes con Python",
            style="Subtitle.TLabel"
        )
        subtitle.pack(pady=(0, 12))

        # Panel de botones
        controls_frame = ttk.Frame(self.root, style="Card.TFrame")
        controls_frame.pack(fill="x", padx=20, pady=8)

        self.btn_load = ttk.Button(
            controls_frame,
            text="Cargar imagen",
            command=self.load_image,
            style="Formal.TButton"
        )
        self.btn_load.grid(row=0, column=0, padx=10, pady=12)

        self.btn_expand = ttk.Button(
            controls_frame,
            text="Aplicar expansión",
            command=self.open_expansion_dialog,
            state=tk.DISABLED,
            style="Formal.TButton"
        )
        self.btn_expand.grid(row=0, column=1, padx=10, pady=12)

        self.btn_equalize = ttk.Button(
            controls_frame,
            text="Aplicar ecualización",
            command=self.apply_equalization,
            state=tk.DISABLED,
            style="Formal.TButton"
        )
        self.btn_equalize.grid(row=0, column=2, padx=10, pady=12)

        self.btn_show_hist = ttk.Button(
            controls_frame,
            text="Mostrar histogramas",
            command=self.show_histograms_window,
            state=tk.DISABLED,
            style="Formal.TButton"
        )
        self.btn_show_hist.grid(row=0, column=3, padx=10, pady=12)

        self.btn_clear = ttk.Button(
            controls_frame,
            text="Limpiar",
            command=self.reset_app,
            style="Formal.TButton"
        )
        self.btn_clear.grid(row=0, column=4, padx=10, pady=12)

        controls_frame.columnconfigure(5, weight=1)

        # Mensaje informativo
        self.info_label = ttk.Label(
            self.root,
            text="Cargue una imagen en escala de grises para iniciar el procesamiento.",
            style="Info.TLabel",
            anchor="center"
        )
        self.info_label.pack(fill="x", padx=20, pady=(5, 8))

        # Contenedor principal de imágenes y matrices
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=(5, 20))

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Panel izquierdo: imagen original + matriz original
        self.left_frame = ttk.Frame(main_frame, style="Card.TFrame")
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Panel derecho: imagen resultante + matriz resultante
        self.right_frame = ttk.Frame(main_frame, style="Card.TFrame")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self.build_image_matrix_panel(
            parent=self.left_frame,
            title="Imagen original cargada",
            image_attr="original_panel",
            matrix_attr="original_matrix_text"
        )

        self.build_image_matrix_panel(
            parent=self.right_frame,
            title="Imagen resultante procesada",
            image_attr="result_panel",
            matrix_attr="result_matrix_text"
        )

    def build_image_matrix_panel(self, parent, title, image_attr, matrix_attr):
        parent.rowconfigure(1, weight=2)
        parent.rowconfigure(3, weight=1)
        parent.columnconfigure(0, weight=1)

        label_title = ttk.Label(
            parent,
            text=title,
            style="CardTitle.TLabel",
            anchor="center"
        )
        label_title.grid(row=0, column=0, sticky="ew", pady=(12, 8))

        image_panel = tk.Label(
            parent,
            bg="#f9fafb",
            relief="groove",
            bd=1
        )
        image_panel.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 10))

        matrix_label = ttk.Label(
            parent,
            text="Matriz de intensidades",
            style="CardTitle.TLabel",
            anchor="center"
        )
        matrix_label.grid(row=2, column=0, sticky="ew", pady=(5, 5))

        matrix_frame = ttk.Frame(parent)
        matrix_frame.grid(row=3, column=0, sticky="nsew", padx=15, pady=(0, 15))
        matrix_frame.rowconfigure(0, weight=1)
        matrix_frame.columnconfigure(0, weight=1)

        matrix_text = tk.Text(
            matrix_frame,
            height=8,
            wrap="none",
            font=("Consolas", 9),
            bg="#f9fafb",
            fg="#111827",
            relief="groove",
            bd=1
        )
        matrix_text.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(matrix_frame, orient="vertical", command=matrix_text.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")

        x_scroll = ttk.Scrollbar(matrix_frame, orient="horizontal", command=matrix_text.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")

        matrix_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        matrix_text.insert("1.0", "Aún no se ha cargado una matriz.")
        matrix_text.config(state=tk.DISABLED)

        setattr(self, image_attr, image_panel)
        setattr(self, matrix_attr, matrix_text)

    def is_grayscale(self, image):
        """
        Verifica si la imagen cargada corresponde a escala de grises.
        Una imagen en escala de grises puede tener una sola dimensión
        o tres canales iguales.
        """
        if len(image.shape) == 2:
            return True

        if len(image.shape) == 3 and image.shape[2] == 3:
            b, g, r = cv2.split(image)
            return np.array_equal(b, g) and np.array_equal(g, r)

        return False

    def calculate_bits(self, image):
        """
        Calcula la cantidad mínima de bits necesaria para representar
        el mayor nivel de intensidad presente en la imagen.
        """
        max_value = int(np.max(image))

        if max_value <= 1:
            return 1

        bits = int(np.ceil(np.log2(max_value + 1)))
        return max(1, min(bits, 8))

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Seleccione una imagen en escala de grises",
            filetypes=[
                ("Archivos de imagen", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Todos los archivos", "*.*")
            ]
        )

        if not file_path:
            return

        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            messagebox.showerror("Error de carga", "No se pudo cargar la imagen seleccionada.")
            return

        if not self.is_grayscale(image):
            messagebox.showerror(
                "Imagen no válida",
                "La imagen ingresada no corresponde a escala de grises.\n\n"
                "Para este proyecto, el programa solo acepta imágenes de un canal "
                "o imágenes cuyos canales RGB tengan la misma intensidad."
            )
            self.reset_app()
            self.info_label.config(
                text="La imagen fue rechazada porque contiene información de color RGB.",
                foreground="#b91c1c"
            )
            return

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.image_gray = image
        self.image_result = image.copy()
        self.image_path = file_path

        self.image_bits = self.calculate_bits(self.image_gray)
        self.total_pixels = self.image_gray.size

        self.display_image(self.image_gray, self.original_panel)
        self.display_image(self.image_result, self.result_panel)

        self.display_matrix(self.image_gray, self.original_matrix_text)
        self.display_matrix(self.image_result, self.result_matrix_text)

        self.enable_buttons()

        self.info_label.config(
            text=(
                f'La imagen cargada es de {self.image_bits} bits '
                f'y tiene {self.total_pixels} píxeles en total. '
                f'Dimensiones: {self.image_gray.shape[1]} x {self.image_gray.shape[0]}. '
                f'Intensidades detectadas: {len(np.unique(self.image_gray))}.'
            ),
            foreground="#047857"
        )

        messagebox.showinfo(
            "Imagen cargada correctamente",
            f'La imagen cargada es de {self.image_bits} bits '
            f'y tiene {self.total_pixels} píxeles en total.'
        )

    def open_expansion_dialog(self):
        if self.image_gray is None:
            messagebox.showwarning("Aviso", "Primero debe cargar una imagen.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Configuración de expansión de histograma")
        dialog.geometry("390x260")
        dialog.configure(bg="#eef2f7")
        dialog.resizable(False, False)

        self.center_window(dialog, 390, 260)

        max_by_bits = (2 ** self.image_bits) - 1

        title = ttk.Label(
            dialog,
            text="Rango de expansión",
            font=("Segoe UI", 14, "bold"),
            background="#eef2f7",
            foreground="#1f2937"
        )
        title.pack(pady=(18, 8))

        explanation = ttk.Label(
            dialog,
            text=(
                "Ingrese el intervalo al que desea expandir las intensidades.\n"
                f"Para una imagen de {self.image_bits} bits, el rango sugerido es [0, {max_by_bits}]."
            ),
            background="#eef2f7",
            foreground="#374151",
            justify="center"
        )
        explanation.pack(pady=5)

        form_frame = ttk.Frame(dialog)
        form_frame.pack(pady=15)

        ttk.Label(form_frame, text="Valor mínimo:").grid(row=0, column=0, padx=8, pady=6)
        min_entry = ttk.Entry(form_frame, width=12)
        min_entry.grid(row=0, column=1, padx=8, pady=6)
        min_entry.insert(0, "0")

        ttk.Label(form_frame, text="Valor máximo:").grid(row=1, column=0, padx=8, pady=6)
        max_entry = ttk.Entry(form_frame, width=12)
        max_entry.grid(row=1, column=1, padx=8, pady=6)
        max_entry.insert(0, str(max_by_bits))

        def confirm_expansion():
            try:
                new_min = int(min_entry.get())
                new_max = int(max_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Los valores del rango deben ser números enteros.")
                return

            if new_min < 0 or new_max < 0:
                messagebox.showerror("Error", "El rango no puede contener valores negativos.")
                return

            if new_min >= new_max:
                messagebox.showerror("Error", "El valor mínimo debe ser menor que el valor máximo.")
                return

            if new_max > 255:
                messagebox.showerror("Error", "El valor máximo no debe superar 255 en imágenes de 8 bits.")
                return

            dialog.destroy()
            self.apply_expansion(new_min, new_max)

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=8)

        ttk.Button(
            button_frame,
            text="Aplicar",
            command=confirm_expansion,
            style="Formal.TButton"
        ).grid(row=0, column=0, padx=8)

        ttk.Button(
            button_frame,
            text="Cancelar",
            command=dialog.destroy,
            style="Formal.TButton"
        ).grid(row=0, column=1, padx=8)

    def apply_expansion(self, new_min, new_max):
        if self.image_gray is None:
            messagebox.showwarning("Aviso", "Primero debe cargar una imagen.")
            return

        img = self.image_gray.astype(np.float32)

        r_min = np.min(img)
        r_max = np.max(img)

        if r_max == r_min:
            messagebox.showwarning(
                "Aviso",
                "No se puede aplicar expansión porque todos los píxeles tienen la misma intensidad."
            )
            return

        expanded = ((img - r_min) * (new_max - new_min) / (r_max - r_min)) + new_min
        expanded = np.round(expanded)
        expanded = np.clip(expanded, 0, 255).astype(np.uint8)

        self.image_result = expanded

        self.display_image(self.image_result, self.result_panel)
        self.display_matrix(self.image_result, self.result_matrix_text)

        self.info_label.config(
            text=(
                f"Expansión de histograma aplicada correctamente. "
                f"Rango original: [{int(r_min)}, {int(r_max)}]. "
                f"Nuevo rango: [{new_min}, {new_max}]."
            ),
            foreground="#1d4ed8"
        )

    def apply_equalization(self):
        if self.image_gray is None:
            messagebox.showwarning("Aviso", "Primero debe cargar una imagen.")
            return

        equalized = cv2.equalizeHist(self.image_gray)

        self.image_result = equalized

        self.display_image(self.image_result, self.result_panel)
        self.display_matrix(self.image_result, self.result_matrix_text)

        self.info_label.config(
            text="Ecualización de histograma aplicada correctamente mediante la función de distribución acumulada.",
            foreground="#1d4ed8"
        )

    def show_histograms_window(self):
        if self.image_gray is None or self.image_result is None:
            messagebox.showwarning("Aviso", "Primero debe cargar y procesar una imagen.")
            return

        hist_window = tk.Toplevel(self.root)
        hist_window.title("Análisis comparativo de histogramas")
        hist_window.geometry("1150x760")
        hist_window.configure(bg="#eef2f7")
        self.center_window(hist_window, 1150, 760)

        title = ttk.Label(
            hist_window,
            text="Comparación de Imagen, Matriz e Histograma",
            font=("Segoe UI", 17, "bold"),
            background="#eef2f7",
            foreground="#1f2937"
        )
        title.pack(pady=(12, 5))

        info = ttk.Label(
            hist_window,
            text=(
                f"Imagen de {self.image_bits} bits | "
                f"Total de píxeles: {self.total_pixels} | "
                f"Dimensiones: {self.image_gray.shape[1]} x {self.image_gray.shape[0]}"
            ),
            font=("Segoe UI", 10),
            background="#eef2f7",
            foreground="#374151"
        )
        info.pack(pady=(0, 8))

        notebook = ttk.Notebook(hist_window)
        notebook.pack(fill="both", expand=True, padx=15, pady=10)

        tab_graphs = ttk.Frame(notebook)
        tab_matrices = ttk.Frame(notebook)

        notebook.add(tab_graphs, text="Imágenes e histogramas")
        notebook.add(tab_matrices, text="Matrices de intensidades")

        # Gráficos con Matplotlib dentro de la ventana
        figure = plt.Figure(figsize=(10.5, 5.8), dpi=100)

        ax1 = figure.add_subplot(2, 2, 1)
        ax1.imshow(self.image_gray, cmap="gray", vmin=0, vmax=255)
        ax1.set_title("Imagen original")
        ax1.axis("off")

        ax2 = figure.add_subplot(2, 2, 2)
        ax2.hist(self.image_gray.ravel(), bins=256, range=(0, 256))
        ax2.set_title("Histograma original")
        ax2.set_xlabel("Nivel de gris")
        ax2.set_ylabel("Frecuencia")

        ax3 = figure.add_subplot(2, 2, 3)
        ax3.imshow(self.image_result, cmap="gray", vmin=0, vmax=255)
        ax3.set_title("Imagen resultante")
        ax3.axis("off")

        ax4 = figure.add_subplot(2, 2, 4)
        ax4.hist(self.image_result.ravel(), bins=256, range=(0, 256))
        ax4.set_title("Histograma resultante")
        ax4.set_xlabel("Nivel de gris")
        ax4.set_ylabel("Frecuencia")

        figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, master=tab_graphs)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # Matrices dentro de la ventana de histogramas
        matrices_frame = ttk.Frame(tab_matrices)
        matrices_frame.pack(fill="both", expand=True, padx=15, pady=15)

        matrices_frame.columnconfigure(0, weight=1)
        matrices_frame.columnconfigure(1, weight=1)
        matrices_frame.rowconfigure(1, weight=1)

        ttk.Label(
            matrices_frame,
            text="Matriz original",
            font=("Segoe UI", 12, "bold")
        ).grid(row=0, column=0, pady=8)

        ttk.Label(
            matrices_frame,
            text="Matriz resultante",
            font=("Segoe UI", 12, "bold")
        ).grid(row=0, column=1, pady=8)

        original_text = self.create_matrix_box(matrices_frame)
        original_text.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        result_text = self.create_matrix_box(matrices_frame)
        result_text.grid(row=1, column=1, sticky="nsew", padx=(8, 0))

        self.insert_matrix_text(original_text, self.image_gray)
        self.insert_matrix_text(result_text, self.image_result)

    def create_matrix_box(self, parent):
        frame = ttk.Frame(parent)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        text = tk.Text(
            frame,
            wrap="none",
            font=("Consolas", 9),
            bg="#f9fafb",
            fg="#111827",
            relief="groove",
            bd=1
        )
        text.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")

        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")

        text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        return frame

    def insert_matrix_text(self, matrix_frame, image):
        text_widget = matrix_frame.winfo_children()[0]
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", self.format_matrix(image))
        text_widget.config(state=tk.DISABLED)

    def display_image(self, image, panel):
        max_size = (500, 310)

        pil_img = Image.fromarray(image)
        pil_img.thumbnail(max_size)

        tk_img = ImageTk.PhotoImage(pil_img)

        panel.config(image=tk_img)
        panel.image = tk_img

    def display_matrix(self, image, text_widget):
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", self.format_matrix(image))
        text_widget.config(state=tk.DISABLED)

    def format_matrix(self, image):
        """
        Devuelve la matriz de intensidades en formato legible.
        También muestra la intensidad mínima y máxima detectada.
        Si la imagen es muy grande, se muestra una vista parcial para evitar saturar la interfaz.
        """
        rows, cols = image.shape

        min_intensity = int(np.min(image))
        max_intensity = int(np.max(image))

        max_rows = 25
        max_cols = 25

        header_text = (
            f"Dimensión de la matriz: {rows} x {cols}\n"
            f"Intensidad mínima detectada: {min_intensity}\n"
            f"Intensidad máxima detectada: {max_intensity}\n\n"
        )

        if rows > max_rows or cols > max_cols:
            preview = image[:max_rows, :max_cols]

            matrix_text = np.array2string(
                preview,
                separator=", ",
                max_line_width=160
            )

            return (
                header_text
                + f"Matriz completa: {rows} x {cols}\n"
                + f"Vista parcial mostrada: {min(rows, max_rows)} x {min(cols, max_cols)}\n\n"
                + f"{matrix_text}\n\n"
                + "Nota: La imagen es grande, por ello se muestra solo una vista parcial "
                + "de la matriz para mantener la legibilidad en pantalla."
            )

        matrix_text = np.array2string(
            image,
            separator=", ",
            max_line_width=160
        )

        return header_text + matrix_text

    def center_window(self, window, width, height):
        window.update_idletasks()

        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))

        window.geometry(f"{width}x{height}+{x}+{y}")

    def clear_panels(self):
        self.original_panel.config(image="")
        self.original_panel.image = None

        self.result_panel.config(image="")
        self.result_panel.image = None

        self.original_matrix_text.config(state=tk.NORMAL)
        self.original_matrix_text.delete("1.0", tk.END)
        self.original_matrix_text.insert("1.0", "Aún no se ha cargado una matriz.")
        self.original_matrix_text.config(state=tk.DISABLED)

        self.result_matrix_text.config(state=tk.NORMAL)
        self.result_matrix_text.delete("1.0", tk.END)
        self.result_matrix_text.insert("1.0", "Aún no se ha generado una matriz resultante.")
        self.result_matrix_text.config(state=tk.DISABLED)

    def enable_buttons(self):
        self.btn_expand.config(state=tk.NORMAL)
        self.btn_equalize.config(state=tk.NORMAL)
        self.btn_show_hist.config(state=tk.NORMAL)

    def disable_buttons(self):
        self.btn_expand.config(state=tk.DISABLED)
        self.btn_equalize.config(state=tk.DISABLED)
        self.btn_show_hist.config(state=tk.DISABLED)

    def reset_app(self):
        self.image_gray = None
        self.image_result = None
        self.image_path = None
        self.image_bits = None
        self.total_pixels = None

        self.clear_panels()
        self.disable_buttons()

        self.info_label.config(
            text="Cargue una imagen en escala de grises para iniciar el procesamiento.",
            foreground="#111827"
        )


def main():
    root = tk.Tk()
    app = HistogramApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()