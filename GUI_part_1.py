import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

class VideoAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Photon Polarization Analysis")
        self.video_path = None
        self.selected_rois = []
        self.bit_entries = []
        self.roi_names = ["Alice-H", "Alice-V", "Bob-H", "Bob-V"]
        self.rois = []
        self.current_frame = None

        # Create the main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()

        # Load video button
        self.load_button = tk.Button(self.main_frame, text="Load Video", command=self.load_video)
        self.load_button.pack()

        # Canvas to display video frame
        self.canvas = tk.Canvas(self.main_frame, width=640, height=480)
        self.canvas.pack()

        # Select ROIs button
        self.roi_button = tk.Button(self.main_frame, text="Select ROIs", command=self.start_roi_selection)
        self.roi_button.pack()

        # Frame to hold bit input fields
        self.bit_frame = tk.Frame(self.main_frame)
        self.bit_frame.pack()

        # Label and entry for number of bits
        tk.Label(self.bit_frame, text="Number of Bits:").grid(row=0, column=0)
        self.num_bits_entry = tk.Entry(self.bit_frame)
        self.num_bits_entry.grid(row=0, column=1)

        # Button to generate bit fields
        self.bit_button = tk.Button(self.bit_frame, text="Generate Bit Fields", command=self.generate_bit_fields)
        self.bit_button.grid(row=1, column=2)

        # Frame for threshold input
        self.threshold_frame = tk.Frame(self.main_frame)
        self.threshold_frame.pack()

        self.threshold_entries = {}

        for idx, roi_name in enumerate(self.roi_names):
            tk.Label(self.threshold_frame, text=f"Threshold for {roi_name}:").grid(row=idx, column=0)
            entry = tk.Entry(self.threshold_frame)
            entry.grid(row=idx, column=1)
            self.threshold_entries[roi_name] = entry

            # Button to visualize intensity over time
            btn = tk.Button(self.threshold_frame, text=f"Show Intensity for {roi_name}",
                            command=lambda roi=roi_name: self.show_intensity(roi))
            btn.grid(row=idx, column=2)

        # Button to start measurement
        self.measure_button = tk.Button(self.threshold_frame, text="Start Measurement", command=self.start_measurement)
        self.measure_button.grid(row=len(self.roi_names), column=1)

    def load_video(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.video_cap = cv2.VideoCapture(filepath)
            if not self.video_cap.isOpened():
                messagebox.showerror("Error", "Unable to open video file")
            else:
                messagebox.showinfo("Success", "Video loaded successfully")
                self.video_path = filepath
                self.show_frame()
        else:
            messagebox.showwarning("Warning", "No file selected")

    def show_frame(self):
        ret, frame = self.video_cap.read()
        if ret:
            self.current_frame = frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk

    def start_roi_selection(self):
        if self.current_frame is None:
            messagebox.showerror("Error", "Please load a video first")
            return

        # Select ROIs for Alice-H, Alice-V, Bob-H, Bob-V
        self.rois = []
        self.roi_idx = 0
        self.rect = None
        self.start_x, self.start_y = 0, 0
        self.selecting = False

        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", self.select_roi)
        self.show_current_frame()

        while len(self.rois) < 4:
            cv2.imshow("Select ROI", self.current_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()

    def select_roi(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x, self.start_y = x, y
            self.selecting = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting:
                self.show_current_frame()
                cv2.rectangle(self.current_frame, (self.start_x, self.start_y), (x, y), (255, 0, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            end_x, end_y = x, y
            roi = (self.start_x, self.start_y, end_x - self.start_x, end_y - self.start_y)
            self.rois.append(roi)
            cv2.rectangle(self.current_frame, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)
            self.roi_idx += 1
            if self.roi_idx < 4:
                messagebox.showinfo("ROI Selection", f"Select ROI for {self.roi_names[self.roi_idx]}")
            else:
                messagebox.showinfo("ROI Selection", "All ROIs selected")

    def show_current_frame(self):
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk

    def generate_bit_fields(self):
        for entry in self.bit_entries:
            entry.destroy()

        self.bit_entries = []
        num_bits = int(self.num_bits_entry.get())

        for i in range(num_bits):
            bit_row = tk.Frame(self.bit_frame)
            bit_row.grid(row=i+1, column=0, columnspan=2)

            tk.Label(bit_row, text=f"Bit {i+1}: Source (0/1):").grid(row=0, column=0)
            source_entry = tk.Entry(bit_row)
            source_entry.grid(row=0, column=1)

            tk.Label(bit_row, text="Start Time (HH:MM:SS):").grid(row=0, column=2)
            start_entry = tk.Entry(bit_row)
            start_entry.grid(row=0, column=3)

            tk.Label(bit_row, text="End Time (HH:MM:SS):").grid(row=0, column=4)
            end_entry = tk.Entry(bit_row)
            end_entry.grid(row=0, column=5)

            self.bit_entries.append((source_entry, start_entry, end_entry))

    def time_to_frame(self, time_str, fps):
        h, m, s = map(int, time_str.split(':'))
        total_seconds = h * 3600 + m * 60 + s
        return int(total_seconds * fps)

    def show_intensity(self, roi_name):
        if not self.video_path:
            messagebox.showerror("Error", "Please load a video first")
            return

        intensities = []
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break

            roi_idx = self.roi_names.index(roi_name)
            x, y, w, h = self.rois[roi_idx]
            roi_frame = frame[y:y+h, x:x+w, 2]  # Red channel
            intensity = np.sum(roi_frame)
            intensities.append(intensity)

        plt.plot(intensities)
        plt.title(f'Intensity Over Time for {roi_name}')
        plt.xlabel('Frame')
        plt.ylabel('Intensity')
        plt.show()

    def start_measurement(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please load a video first")
            return

        thresholds = {roi_name: int(entry.get()) for roi_name, entry in self.threshold_entries.items()}
        num_bits = int(self.num_bits_entry.get())
        bit_data = []
        fps = int(self.video_cap.get(cv2.CAP_PROP_FPS))

        for bit_idx in range(num_bits):
            source_entry, start_entry, end_entry = self.bit_entries[bit_idx]
            source = int(source_entry.get())
            if source not in [0, 1]:
                messagebox.showerror("Error", "Source must be 0 or 1")
                return

            start_time = self.time_to_frame(start_entry.get(), fps)
            end_time = self.time_to_frame(end_entry.get(), fps)

            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_time)
            intensities = {name: 0 for name in self.roi_names}

            for frame_idx in range(start_time, end_time + 1):
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                for roi_idx, (x, y, w, h) in enumerate(self.rois):
                    roi_frame = frame[y:y+h, x:x+w, 2]  # Red channel
                    intensity = np.sum(roi_frame)
                    intensities[self.roi_names[roi_idx]] += intensity

            bit_data.append((source, intensities))

        self.construct_density_matrix(bit_data, thresholds)

    def construct_density_matrix(self, bit_data, thresholds):
        # Construct the density matrix
        matrix_size = len(self.roi_names)
        density_matrix = np.zeros((matrix_size, matrix_size))

        for source, intensities in bit_data:
            alice_h_intensity = intensities["Alice-H"]
            alice_v_intensity = intensities["Alice-V"]
            bob_h_intensity = intensities["Bob-H"]
            bob_v_intensity = intensities["Bob-V"]

            if source == 0:  # Source 0 corresponds to "HH"
                if alice_h_intensity >= thresholds["Alice-H"] and bob_h_intensity >= thresholds["Bob-H"]:
                    density_matrix[0, 0] += 1  # HH
                if alice_h_intensity >= thresholds["Alice-H"] and bob_v_intensity >= thresholds["Bob-V"]:
                    density_matrix[0, 1] += 1  # HV
                if alice_v_intensity >= thresholds["Alice-V"] and bob_h_intensity >= thresholds["Bob-H"]:
                    density_matrix[0, 2] += 1  # VH
                if alice_v_intensity >= thresholds["Alice-V"] and bob_v_intensity >= thresholds["Bob-V"]:
                    density_matrix[0, 3] += 1  # VV

            elif source == 1:  # Source 1 corresponds to "VV"
                if alice_h_intensity >= thresholds["Alice-H"] and bob_h_intensity >= thresholds["Bob-H"]:
                    density_matrix[3, 0] += 1  # HH
                if alice_h_intensity >= thresholds["Alice-H"] and bob_v_intensity >= thresholds["Bob-V"]:
                    density_matrix[3, 1] += 1  # HV
                if alice_v_intensity >= thresholds["Alice-V"] and bob_h_intensity >= thresholds["Bob-H"]:
                    density_matrix[3, 2] += 1  # VH
                if alice_v_intensity >= thresholds["Alice-V"] and bob_v_intensity >= thresholds["Bob-V"]:
                    density_matrix[3, 3] += 1  # VV

        density_matrix /= np.sum(density_matrix)
        self.print_density_matrix(density_matrix)
        self.visualize_density_matrix(density_matrix)

    def print_density_matrix(self, density_matrix):
        print("Density Matrix:")
        print(density_matrix)

    def visualize_density_matrix(self, density_matrix):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_labels = ["HH", "HV", "VH", "VV"]
        y_labels = ["HH", "HV", "VH", "VV"]
        xpos, ypos = np.meshgrid(range(density_matrix.shape[0]), range(density_matrix.shape[1]))

        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)

        dx = dy = 0.8
        dz = density_matrix.flatten()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.set_zlabel('Density')
        ax.set_xlabel('Source')
        ax.set_ylabel('Detectors')
        ax.set_title('Density Matrix Visualization')

        plt.show()

# Initialize the main application
root = tk.Tk()
app = VideoAnalysisApp(root)
root.mainloop()
