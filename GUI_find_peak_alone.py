import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class VideoAnalysisApp:
	def __init__(self, root):
		self.root = root
		self.root.title("Photon Polarization Analysis")
		self.video_path = None
		self.selected_rois = []
		self.roi_names = ["Alice-H", "Alice-V", "Bob-H", "Bob-V"]
		self.rois = []
		self.current_frame = None
		self.peaks = {roi_name: {"H": [], "V": []} for roi_name in self.roi_names}

		# Create the main frame
		self.main_frame = tk.Frame(self.root)
		self.main_frame.pack()

		# Load video button
		self.load_button = tk.Button(self.main_frame, text="Load Video", command=self.load_video)
		self.load_button.pack()

		# Canvas to display video frame
		self.canvas = tk.Canvas(self.main_frame, width=640, height=400)
		self.canvas.pack()

		# Select ROIs button
		self.roi_button = tk.Button(self.main_frame, text="Select ROIs", command=self.start_roi_selection)
		self.roi_button.pack()

		# Frame for threshold input
		self.threshold_frame = tk.Frame(self.main_frame)
		self.threshold_frame.pack()

		self.threshold_entries = {}

		for idx, roi_name in enumerate(self.roi_names):
			tk.Label(self.threshold_frame, text=f"Threshold for {roi_name}:").grid(row=idx, column=0)
			entry = tk.Entry(self.threshold_frame)
			entry.grid(row=idx, column=1)
			self.threshold_entries[roi_name] = entry

			# Button to visualize intensity over time and mark peaks
			btn = tk.Button(self.threshold_frame, text=f"Show Intensity for {roi_name}",
							command=lambda roi=roi_name: self.show_intensity(roi))
			btn.grid(row=idx, column=2)

		# Entry for number of 1s (V) and 0s (H)
		tk.Label(self.threshold_frame, text="Number of 1s (V):").grid(row=len(self.roi_names), column=0)
		self.num_v_entry = tk.Entry(self.threshold_frame)
		self.num_v_entry.grid(row=len(self.roi_names), column=1)

		tk.Label(self.threshold_frame, text="Number of 0s (H):").grid(row=len(self.roi_names) + 1, column=0)
		self.num_h_entry = tk.Entry(self.threshold_frame)
		self.num_h_entry.grid(row=len(self.roi_names) + 1, column=1)

		# Button to start measurement
		self.measure_button = tk.Button(self.threshold_frame, text="Start Measurement", command=self.start_measurement)
		self.measure_button.grid(row=len(self.roi_names) + 2, column=1)

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

		# Normalize intensities
		intensities = np.array(intensities)
		intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())

		# Find peaks
		num_v = int(self.num_v_entry.get())
		num_h = int(self.num_h_entry.get())

		peaks, _ = find_peaks(intensities, height=None, prominence=0.5)

		# Sort peaks by intensity (highest first)
		sorted_peaks = sorted(peaks, key=lambda x: intensities[x], reverse=True)

		# Take top num_v peaks as V and next num_h peaks as H
		peaks_v = sorted_peaks[:num_v]
		peaks_h = sorted_peaks[num_v:num_v + num_h]

		# Mark peaks on intensity plot
		plt.figure()
		plt.plot(intensities)
		plt.title(f'Intensity Over Time for {roi_name}')
		plt.xlabel('Frame')
		plt.ylabel('Intensity')
		plt.scatter(peaks_v, [intensities[i] for i in peaks_v], color='r', label='Peak V')
		plt.scatter(peaks_h, [intensities[i] for i in peaks_h], color='g', label='Peak H')
		plt.legend()
		plt.show()

		# Store peaks
		self.peaks[roi_name]["H"] = peaks_h
		self.peaks[roi_name]["V"] = peaks_v

	def start_measurement(self):
		if not self.video_path:
			messagebox.showerror("Error", "Please load a video first")
			return

		thresholds = {roi_name: float(entry.get()) for roi_name, entry in self.threshold_entries.items()}
		num_v = int(self.num_v_entry.get())
		num_h = int(self.num_h_entry.get())

		# Create lists of frame positions for peaks
		alice_h_frames = self.peaks["Alice-H"]["H"]
		alice_v_frames = self.peaks["Alice-V"]["V"]
		bob_h_frames = self.peaks["Bob-H"]["H"]
		bob_v_frames = self.peaks["Bob-V"]["V"]

		# Check tolerance function
		def check_tolerance(alice_frames, bob_frames):
			return all(abs(a - b) < 10 for a, b in zip(sorted(alice_frames), sorted(bob_frames)))

		if not (check_tolerance(alice_h_frames, bob_h_frames) and check_tolerance(alice_v_frames, bob_v_frames)):
			messagebox.showerror("Error", "Peaks do not match within tolerance")
			return

		# Merge lists
		alice_merge = sorted(set(alice_h_frames + alice_v_frames))
		bob_merge = sorted(set(bob_h_frames + bob_v_frames))

		# Calculate intensities for all frames
		intensities = []
		self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
		while True:
			ret, frame = self.video_cap.read()
			if not ret:
				break

			intensity = self.calculate_intensity(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES) - 1, self.rois[0])
			intensities.append(intensity)

		# Normalize intensities
		intensities = np.array(intensities)
		intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())

		# Create intensity pairs
		alice_intensity_pairs = [(intensities[frame],) for frame in alice_merge]
		bob_intensity_pairs = [(intensities[frame],) for frame in bob_merge]

		# Construct the density matrix
		density_matrix = np.zeros((4, 4))

		for j in range(len(alice_intensity_pairs)):
			alice_intensity = alice_intensity_pairs[j]
			bob_intensity = bob_intensity_pairs[j]

			if bob_intensity > thresholds["Bob-H"] and alice_intensity > thresholds["Alice-H"]:
				density_matrix[0, 0] += 1
				density_matrix[0, 3] += 1

			if bob_intensity > thresholds["Bob-V"] and alice_intensity > thresholds["Alice-V"]:
				density_matrix[3, 3] += 1
				density_matrix[3, 0] += 1

		print("Density Matrix:\n", density_matrix)


if __name__ == "__main__":
	root = tk.Tk()
	app = VideoAnalysisApp(root)
	root.mainloop()