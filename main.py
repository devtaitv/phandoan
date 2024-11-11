import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os


class EnhancedImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Xử lý Ảnh Vệ tinh Nâng cao")
        self.root.attributes("-zoomed", True)

        # Biến lưu trữ
        self.original_image = None
        self.processed_image = None
        self.current_image = None
        self.history = []
        self.history_index = -1

        # Biến lưu trữ PhotoImage
        self.original_photo = None
        self.processed_photo = None

        # Style
        self.style = ttk.Style()
        self.style.configure('Custom.TButton', padding=5)
        self.style.configure('Custom.TFrame', background='#f0f0f0')

        self.create_gui()

    def create_gui(self):
        # Main container
        main_container = ttk.Frame(self.root, style='Custom.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for controls
        left_panel = ttk.Frame(main_container, style='Custom.TFrame')
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # File operations
        file_frame = ttk.LabelFrame(left_panel, text="Thao tác tệp", padding=5)
        file_frame.pack(fill=tk.X, pady=5)

        ttk.Button(file_frame, text="Tải ảnh", command=self.load_image, style='Custom.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Lưu ảnh", command=self.save_image, style='Custom.TButton').pack(fill=tk.X, pady=2)

        # History operations
        history_frame = ttk.LabelFrame(left_panel, text="Lịch sử", padding=5)
        history_frame.pack(fill=tk.X, pady=5)

        ttk.Button(history_frame, text="Hoàn tác", command=self.undo, style='Custom.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(history_frame, text="Làm lại", command=self.redo, style='Custom.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(history_frame, text="Khôi phục gốc", command=self.reset_image, style='Custom.TButton').pack(
            fill=tk.X, pady=2)

        # Edge detection
        edge_frame = ttk.LabelFrame(left_panel, text="Phát hiện cạnh", padding=5)
        edge_frame.pack(fill=tk.X, pady=5)

        ttk.Button(edge_frame, text="Sobel", command=self.apply_sobel, style='Custom.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(edge_frame, text="Prewitt", command=self.apply_prewitt, style='Custom.TButton').pack(fill=tk.X,
                                                                                                        pady=2)
        ttk.Button(edge_frame, text="Roberts", command=self.apply_roberts, style='Custom.TButton').pack(fill=tk.X,
                                                                                                        pady=2)
        ttk.Button(edge_frame, text="Canny", command=self.apply_canny, style='Custom.TButton').pack(fill=tk.X, pady=2)

        # Image enhancement
        enhance_frame = ttk.LabelFrame(left_panel, text="Tăng cường ảnh", padding=5)
        enhance_frame.pack(fill=tk.X, pady=5)

        # Brightness control
        ttk.Label(enhance_frame, text="Độ sáng:").pack()
        self.brightness_var = tk.IntVar(value=0)
        ttk.Scale(enhance_frame, from_=-100, to=100, variable=self.brightness_var,
                  command=self.apply_brightness).pack(fill=tk.X)

        # Contrast control
        ttk.Label(enhance_frame, text="Độ tương phản:").pack()
        self.contrast_var = tk.DoubleVar(value=1.0)
        ttk.Scale(enhance_frame, from_=0.1, to=3.0, variable=self.contrast_var,
                  command=self.apply_contrast).pack(fill=tk.X)

        # Gaussian Blur
        ttk.Label(enhance_frame, text="Gaussian Blur:").pack()
        self.gaussian_var = tk.IntVar(value=1)
        ttk.Scale(enhance_frame, from_=1, to=21, variable=self.gaussian_var,
                  command=self.apply_gaussian).pack(fill=tk.X)

        # Thresholding
        thresh_frame = ttk.LabelFrame(left_panel, text="Phân ngưỡng", padding=5)
        thresh_frame.pack(fill=tk.X, pady=5)

        ttk.Label(thresh_frame, text="Ngưỡng:").pack()
        self.threshold_var = tk.IntVar(value=127)
        ttk.Scale(thresh_frame, from_=0, to=255, variable=self.threshold_var,
                  command=self.apply_threshold).pack(fill=tk.X)

        # Advanced operations
        advanced_frame = ttk.LabelFrame(left_panel, text="Xử lý nâng cao", padding=5)
        advanced_frame.pack(fill=tk.X, pady=5)

        ttk.Button(advanced_frame, text="Phân đoạn vùng", command=self.apply_watershed,
                   style='Custom.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(advanced_frame, text="Phát hiện đường thẳng", command=self.detect_lines,
                   style='Custom.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(advanced_frame, text="Phát hiện góc", command=self.detect_corners,
                   style='Custom.TButton').pack(fill=tk.X, pady=2)

        # Right panel for image display
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Image information
        self.info_label = ttk.Label(right_panel, text="")
        self.info_label.pack()

        # Create image display frame
        display_frame = ttk.Frame(right_panel)
        display_frame.pack(fill=tk.BOTH, expand=True)

        # Original image canvas (left side)
        self.original_canvas = tk.Canvas(display_frame, bg='#e0e0e0')
        self.original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(self.original_canvas, text="Ảnh gốc").pack()

        # Separator
        ttk.Separator(display_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Processed image canvas (right side)
        self.processed_canvas = tk.Canvas(display_frame, bg='#e0e0e0')
        self.processed_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ttk.Label(self.processed_canvas, text="Ảnh đã xử lý").pack()

        # Bind canvas resize events
        self.original_canvas.bind('<Configure>', self.on_canvas_resize)
        self.processed_canvas.bind('<Configure>', self.on_canvas_resize)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
                return
            self.current_image = self.original_image.copy()
            self.history = [self.original_image.copy()]
            self.history_index = 0
            self.update_display()
            self.update_info()

    def save_image(self):
        if self.current_image is None:
            messagebox.showwarning("Cảnh báo", "Không có ảnh để lưu!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"),
                       ("PNG files", "*.png"),
                       ("All files", "*.*")])

        if file_path:
            cv2.imwrite(file_path, self.current_image)
            messagebox.showinfo("Thông báo", "Đã lưu ảnh thành công!")

    def update_display(self):
        if self.original_image is None:
            return

        # Get canvas dimensions
        canvas_width = self.original_canvas.winfo_width()
        canvas_height = self.original_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        # Display original image
        original_display = self.original_image.copy()
        if len(original_display.shape) == 2:
            original_display = cv2.cvtColor(original_display, cv2.COLOR_GRAY2RGB)
        else:
            original_display = cv2.cvtColor(original_display, cv2.COLOR_BGR2RGB)

        # Display processed image
        processed_display = self.current_image.copy() if self.current_image is not None else original_display
        if len(processed_display.shape) == 2:
            processed_display = cv2.cvtColor(processed_display, cv2.COLOR_GRAY2RGB)
        else:
            processed_display = cv2.cvtColor(processed_display, cv2.COLOR_BGR2RGB)

        # Calculate scaling
        scale = min(canvas_width / original_display.shape[1],
                    canvas_height / original_display.shape[0])
        new_width = int(original_display.shape[1] * scale)
        new_height = int(original_display.shape[0] * scale)

        if new_width > 0 and new_height > 0:
            # Resize images
            original_display = cv2.resize(original_display, (new_width, new_height))
            processed_display = cv2.resize(processed_display, (new_width, new_height))

            # Convert to PhotoImage
            self.original_photo = ImageTk.PhotoImage(image=Image.fromarray(original_display))
            self.processed_photo = ImageTk.PhotoImage(image=Image.fromarray(processed_display))

            # Update canvases
            self.original_canvas.delete("all")
            self.processed_canvas.delete("all")

            self.original_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=self.original_photo, anchor=tk.CENTER)
            self.processed_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=self.processed_photo, anchor=tk.CENTER)

    def update_info(self):
        if self.current_image is None:
            self.info_label.config(text="")
            return

        height, width = self.current_image.shape[:2]
        channels = self.current_image.shape[2] if len(self.current_image.shape) > 2 else 1
        size_mb = os.path.getsize(self.temp_file()) / (1024 * 1024) if self.temp_file() else 0

        info_text = f"Kích thước: {width}x{height} | Kênh màu: {channels} | "
        info_text += f"Dung lượng: {size_mb:.1f}MB"
        self.info_label.config(text=info_text)

    def temp_file(self):
        if self.current_image is None:
            return None
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, self.current_image)
        return temp_path

    def add_to_history(self, image):
        self.history_index += 1
        self.history = self.history[:self.history_index]
        self.history.append(image.copy())
        self.current_image = image.copy()
        self.update_display()
        self.update_info()

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.current_image = self.history[self.history_index].copy()
            self.update_display()

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_image = self.history[self.history_index].copy()
            self.update_display()

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.add_to_history(self.current_image)

    def apply_sobel(self):
        if self.original_image is None:
            return

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        magnitude = np.uint8(magnitude)

        self.add_to_history(cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR))

    def apply_prewitt(self):
        if self.original_image is None:
            return

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

        prewitt_x = cv2.filter2D(gray, -1, kernelx)
        prewitt_y = cv2.filter2D(gray, -1, kernely)

        magnitude = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)
        magnitude = np.uint8(magnitude)

        self.add_to_history(cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR))

    def apply_roberts(self):
        if self.current_image is None:
            return

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        roberts_x = np.array([[1, 0], [0, -1]])
        roberts_y = np.array([[0, 1], [-1, 0]])

        gx = cv2.filter2D(gray, -1, roberts_x)
        gy = cv2.filter2D(gray, -1, roberts_y)

        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        magnitude = np.uint8(magnitude)

        self.add_to_history(cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR))

    def apply_canny(self):
        if self.current_image is None:
            return

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        self.add_to_history(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    def apply_gaussian(self, event=None):
        if self.current_image is None:
            return

        kernel_size = self.gaussian_var.get()
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred = cv2.GaussianBlur(self.current_image, (kernel_size, kernel_size), 0)
        self.add_to_history(blurred)

    def apply_brightness(self, event=None):
        if self.current_image is None:
            return

        brightness = self.brightness_var.get()
        adjusted = self.current_image.copy()

        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness

        alpha = (highlight - shadow) / 255
        gamma = shadow

        adjusted = cv2.addWeighted(adjusted, alpha, adjusted, 0, gamma)
        self.add_to_history(adjusted)

    def apply_contrast(self, event=None):
        if self.current_image is None:
            return

        contrast = self.contrast_var.get()
        adjusted = cv2.convertScaleAbs(self.current_image, alpha=contrast)
        self.add_to_history(adjusted)

    def apply_threshold(self, event=None):
        if self.current_image is None:
            return

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        thresh_value = self.threshold_var.get()
        _, thresholded = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
        self.add_to_history(cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR))

    def apply_watershed(self):
        if self.current_image is None:
            return

        # Chuyển sang grayscale và áp dụng threshold
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Apply watershed
        markers = cv2.watershed(self.current_image, markers)

        # Visualize results
        result = self.current_image.copy()
        result[markers == -1] = [0, 0, 255]  # Mark watershed boundaries in red

        self.add_to_history(result)

    def detect_lines(self):
        if self.current_image is None:
            return

        # Convert to grayscale
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # HoughLines detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=10)

        # Draw lines
        result = self.current_image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.add_to_history(result)

    def detect_corners(self):
        if self.current_image is None:
            return

        # Convert to grayscale
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

        # Detect corners using Harris Corner Detection
        corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        # Dilate corner detections
        corners = cv2.dilate(corners, None)

        # Create result image
        result = self.current_image.copy()

        # Mark corners with red color
        result[corners > 0.01 * corners.max()] = [0, 0, 255]

        self.add_to_history(result)

    def on_canvas_resize(self, event):
            self.update_display()

    def run(self):
            self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedImageProcessingApp(root)
    app.run()
