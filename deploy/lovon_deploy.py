import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import numpy as np
import pyrealsense2 as rs
import time
import torch
import threading
import queue
import argparse
from ultralytics import YOLO
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient as Go2VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient as Go2SportClient
from unitree_sdk2py.h1.loco.h1_loco_client import LocoClient as H1SportClient
from unitree_sdk2py.b2.sport.sport_client import SportClient as B2SportClient
from unitree_sdk2py.b2.front_video.front_video_client import FrontVideoClient as B2FrontVideoClient
from unitree_sdk2py.b2.back_video.back_video_client import BackVideoClient as B2BackVideoClient
# from cxn_010.api_object_extraction_transformer import ObjectExtractionAPI
# from cxn_010.api_language2motion_transformer import MotionPredictor

from models.api_object_extraction import SequenceToSequenceClassAPI
from models.api_language2mostion import MotionPredictor

from tkinter import Tk, Entry, Button, Label, Frame
from PIL import Image, ImageTk
import cv2

import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ImageGetterThread(threading.Thread):
    """Image Acquisition Thread"""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.running = True
        self.image_queue = queue.Queue(maxsize=1)  # Keep only the latest frame
        self.freq_start = time.time()
        self.freq_count = 0

    def run(self):
        while self.running:
            try:
                if self.controller.camera_type == "inner":
                    self.controller._update_image_from_video_client()
                elif self.controller.camera_type == "realsense":
                    self.controller._update_image_from_realsense()

                # Ensure only the latest frame is kept in the queue
                with self.controller.image_lock:
                    if hasattr(self.controller, 'image') and self.controller.image is not None:
                        current_image = self.controller.image.copy()
                        if not self.image_queue.empty():
                            try:
                                self.image_queue.get_nowait()
                            except queue.Empty:
                                pass
                        # Calculate Laplacian variance to detect blur
                        laplacian_var, is_blur = self.detect_blur(
                            current_image, threshold=self.controller.blur_threshold
                        )
                        if not is_blur:
                            # Only put clear images into the queue
                            self.image_queue.put(current_image)
                            self.last_image = current_image
                        else:
                            # Discard blurry image and use the last clear one if available
                            if hasattr(self, 'last_image'):
                                self.image_queue.put(self.last_image)
                            else:
                                print("No clear image available to use as fallback.")

                self.freq_count += 1
                if time.time() - self.freq_start >= 1:
                    freq = self.freq_count / (time.time() - self.freq_start)
                    print(f"[ImageGetter] Frequency: {freq:.2f} Hz")
                    with self.controller.freq_lock:
                        self.controller.image_getter_freq = freq
                    self.freq_start = time.time()
                    self.freq_count = 0

            except Exception as e:
                print(f"ImageGetter Error: {e}")
                time.sleep(0.1)

    @staticmethod
    def detect_blur(image, threshold=100.0):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (measure of sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Determine if image is blurry
        is_blur = laplacian_var < threshold
        
        # Print detection result
        print(f"Laplacian Variance: {laplacian_var:.2f}")
        print("Image is blurry, discarding" if is_blur else "Image is clear")
        return laplacian_var, is_blur
    
    def stop(self):
        self.running = False
        self.join()


class YoloProcessingThread(threading.Thread):
    """YOLO Object Detection Processing Thread"""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.running = True
        self.image_queue = controller.image_getter_thread.image_queue
        self.result_queue = queue.Queue()
        self.freq_start = time.time()
        self.freq_count = 0

    def run(self):
        while self.running:
            try:
                image = self.image_queue.get(timeout=1)
                with self.controller.yolo_lock:
                    results = self.controller.yolo_model(image)
                    # Pass original image for post-processing (mainly for dimension retrieval)
                    self.controller._yolo_image_post_process(results, image)

                self.result_queue.put(self.controller.state.copy())

                self.freq_count += 1
                if time.time() - self.freq_start >= 1:
                    freq = self.freq_count / (time.time() - self.freq_start)
                    print(f"[YoloProcessor] Frequency: {freq:.2f} Hz")
                    with self.controller.freq_lock:
                        self.controller.yolo_processor_freq = freq
                    self.freq_start = time.time()
                    self.freq_count = 0

            except queue.Empty:
                continue
            except Exception as e:
                print(f"YoloProcessing Error: {e}")
                time.sleep(0.1)

    def stop(self):
        self.running = False
        self.join()


class MotionControlThread(threading.Thread):
    """Robot Motion Control Thread"""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.running = True
        self.result_queue = controller.yolo_processing_thread.result_queue
        self.freq_start = time.time()
        self.freq_count = 0

    def run(self):
        while self.running:
            try:
                state = self.result_queue.get(timeout=1)
                with self.controller.motion_lock:
                    self.controller._update_motion_control(state)
                    self.controller._control_robot()

                self.freq_count += 1
                if time.time() - self.freq_start >= 1:
                    freq = self.freq_count / (time.time() - self.freq_start)
                    print(f"[MotionControl] Frequency: {freq:.2f} Hz")
                    with self.controller.freq_lock:
                        self.controller.motion_control_freq = freq
                    self.freq_start = time.time()
                    self.freq_count = 0

            except queue.Empty:
                continue
            except Exception as e:
                print(f"MotionControl Error: {e}")
                time.sleep(0.1)

    def stop(self):
        self.running = False
        self.join()


class VisualLanguageController:
    def __init__(self, yolo_model_dir="yolo-models/yolo11n.pt", 
                 tokenizer_path=None, 
                 object_extraction_model_path=None, 
                 language2motion_model_path=None,
                 camera_type='inner', 
                 robot_type='go2', 
                 show_video=True, 
                 show_max_result=False,
                 show_arrowed=False,
                 blur_threshold=10.0,
                 lengthen_filter=3):
        # Initialize core functional components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.object_extractor = SequenceToSequenceClassAPI(
            model_path=object_extraction_model_path,
            tokenizer_path=tokenizer_path
        )
        self.yolo_model = YOLO(yolo_model_dir)
        self.motion_predictor = MotionPredictor(
            model_path=language2motion_model_path,
            tokenizer_path=tokenizer_path
        )

        # Command-line configurable parameters
        self.show_video = show_video
        self.show_max_result = show_max_result
        self.show_arrowed = show_arrowed  # Whether to display arrow vectors
        self.camera_type = camera_type
        self.robot_type = robot_type
        self.blur_threshold = blur_threshold  # Threshold for blur detection
        self.lengthen_filter = lengthen_filter  # Number of historical detection results to keep

        self.button_update_inst = False

        # Initialize RealSense camera if selected
        if self.camera_type == "realsense":
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
            self.pipeline.start(self.config)

        # Initialize Unitree SDK components
        self._init_channel_factory()
        self.video_client = self._init_camera()
        self.sport_client = self._init_sport()

        # Initialize mission instructions and state
        self.mission_instruction_0 = "run to the person at speed of 0.36 m/s"
        self.mission_instruction_1 = self.mission_instruction_0
        self.state = {
            "predicted_object": "NULL",
            "confidence": [0.00],
            "object_xyn": [0.00, 0.00],
            "object_whn": [0.00, 0.00],
            "mission_state_in": "success",
            "search_state_in": "had_searching_1",
        }
        self.extracted_object = self.object_extractor.predict(self.mission_instruction_1)

        # Initialize thread locks
        self.image_lock = threading.Lock()
        self.yolo_lock = threading.Lock()
        self.motion_lock = threading.Lock()
        self.freq_lock = threading.Lock()  # Lock for frequency updates

        # Initialize frequency monitoring variables
        self.image_getter_freq = 0.0
        self.yolo_processor_freq = 0.0
        self.motion_control_freq = 0.0

        # Initialize worker threads
        self.image_getter_thread = ImageGetterThread(self)
        self.yolo_processing_thread = YoloProcessingThread(self)
        self.motion_control_thread = MotionControlThread(self)

        # Initialize UI
        self.root = Tk()
        self.root.title("Visual Language Motion Controller")
        self.font_style = ("Arial", 16, "bold")
        self.small_font = ("Arial", 14, "bold")  # Font for frequency display

        # Create left (image) and right (instruction) frames
        self.image_frame = Frame(self.root)
        self.image_frame.pack(side='left', fill='both', expand=True)

        self.instruction_frame = Frame(self.root)
        self.instruction_frame.pack(side='right', fill='both', expand=True)

        self.init_ui()

        # Initialize video display if enabled
        if show_video:
            self.image_label = Label(self.image_frame)
            self.image_label.pack(fill='both', expand=True)
            self.update_image()

    def init_ui(self):
        """Initialize UI Interface"""
        screen_width = self.root.winfo_screenwidth()
        window_width = 1800
        window_height = 1000
        # Set window position (right-aligned) and size
        self.root.geometry(f"{window_width}x{window_height}+{screen_width - 1850}+20")

        # Robot control buttons (top of right frame)
        control_frame = Frame(self.instruction_frame)
        control_frame.pack(pady=10, padx=10, anchor='n')
        Button(control_frame, text="Damp", command=self.sport_client.Damp,
               font=self.font_style, width=15).pack(side='left', padx=5)

        # Mission instruction input area
        initial_instructions = [
            "Run to the bus at speed of 0.36 m/s",
            "move to the person at speed of 0.7 m/s",
            "Run to the human at speed of 0.5 m/s",
            "run to the chair at speed of 0.4 m/s",
            "approach the car at speed of 0.5 m/s",
            "run to the bicycle at speed of 0.4 m/s",
            "Rush to the chair at speed of 0.3 m/s",
            "move to the armchair at speed of 0.35 m/s",
            "Sprint to the game ball at speed of 0.35 m/s",
            "Approach to the laptop at speed of 0.3 m/s"
        ]

        self.instruction_entries = []
        for idx, instr in enumerate(initial_instructions):
            entry = Entry(self.instruction_frame, width=60, font=self.font_style)
            entry.insert(0, instr)
            entry.pack(pady=5)
            self.instruction_entries.append(entry)

            # Button to submit current instruction
            button = Button(
                self.instruction_frame,
                text=f"Submit Mission {idx + 1}",
                command=lambda e=entry: self.update_instruction(e),
                font=self.font_style
            )
            button.pack(pady=2)

        # Status display labels
        self.mission_label = Label(self.instruction_frame, text="Current Mission: ", font=self.font_style)
        self.mission_label.pack(pady=10)
        self.object_label = Label(self.instruction_frame, text="Extracted Object: ", font=self.font_style)
        self.object_label.pack(pady=10)
        self.state_label = Label(self.instruction_frame, text="Mission State: ", font=self.font_style)
        self.state_label.pack(pady=10)
        self.motion_label = Label(self.instruction_frame, text="Motion Vector: ", font=self.font_style)
        self.motion_label.pack(pady=10)

        # Frequency display area (bottom of right frame)
        freq_display_frame = Frame(self.instruction_frame, bd=1, relief='sunken', padx=10, pady=5)
        freq_display_frame.pack(side='bottom', fill='both', expand=True, padx=10, pady=10)

        self.freq_image_label = Label(freq_display_frame, text="[ImageGetter] Frequency: 0.00 Hz", 
                                      font=self.small_font, anchor='w', fg='red')
        self.freq_image_label.pack(anchor='w', pady=2)

        self.freq_yolo_label = Label(freq_display_frame, text="[YoloProcessor] Frequency: 0.00 Hz", 
                                     font=self.small_font, anchor='w', fg='red')
        self.freq_yolo_label.pack(anchor='w', pady=2)

        self.freq_motion_label = Label(freq_display_frame, text="[MotionControl] Frequency: 0.00 Hz", 
                                       font=self.small_font, anchor='w', fg='red')
        self.freq_motion_label.pack(anchor='w', pady=2)

        self.update_ui_labels()

    def update_ui_labels(self):
        """Update UI Status Labels"""
        self.mission_label.config(text=f"Current Mission: {self.mission_instruction_1}")
        self.object_label.config(text=f"Extracted Object: {self.extracted_object}")
        
        # Update mission instruction history
        if self.button_update_inst:
            self.button_update_inst = False
        else:
            self.mission_instruction_0 = self.mission_instruction_1
        
        self.state_label.config(text=f"Mission State: {self.state['mission_state_in']}")
        motion_text = f"Motion Vector: {self.motion_vector}" if hasattr(self, 'motion_vector') else "Motion Vector: Not Available"
        self.motion_label.config(text=motion_text)
        
        # Refresh every 1 second
        self.root.after(1000, self.update_ui_labels)

    def update_freq_display(self):
        """Update Frequency Display Labels"""
        with self.freq_lock:
            img_freq = f"{self.image_getter_freq:.2f}"
            yolo_freq = f"{self.yolo_processor_freq:.2f}"
            motion_freq = f"{self.motion_control_freq:.2f}"
        
        self.freq_image_label.config(text=f"[ImageGetter] Frequency: {img_freq} Hz")
        self.freq_yolo_label.config(text=f"[YoloProcessor] Frequency: {yolo_freq} Hz")
        self.freq_motion_label.config(text=f"[MotionControl] Frequency: {motion_freq} Hz")
        
        # Refresh every 100ms
        self.root.after(100, self.update_freq_display)

    def update_instruction(self, entry):
        """Process Mission Instruction Submission"""
        new_instr = entry.get()
        if new_instr:
            # Update instruction history
            self.mission_instruction_0 = self.mission_instruction_1
            self.mission_instruction_1 = new_instr
            # Extract target object from new instruction
            self.extracted_object = self.object_extractor.predict(new_instr)
            self.button_update_inst = True
            print(f"Updated Mission Instruction: {self.mission_instruction_1}")
            self.update_ui_labels()  # Immediately update UI

    def _init_channel_factory(self):
        """Initialize Unitree Channel Factory"""
        if len(sys.argv) > 1:
            ChannelFactoryInitialize(0, sys.argv[1])
        else:
            ChannelFactoryInitialize(0)

    def _init_camera(self):
        """Initialize Robot Camera Client Based on Robot Type"""
        if self.robot_type == "go2":
            client = Go2VideoClient()
        elif self.robot_type == "h1":
            client = Go2VideoClient()
        elif self.robot_type == "b2":
            client = B2FrontVideoClient()
        else:
            raise ValueError("Unsupported robot type. Supported types are: go2, h1, b2.")
        client.SetTimeout(3.0)
        client.Init()
        return client

    def _init_sport(self):
        """Initialize Robot Motion Control Client Based on Robot Type"""
        if self.robot_type == "go2":
            sport_client = Go2SportClient()
        elif self.robot_type == "h1":
            sport_client = H1SportClient()
        elif self.robot_type == "b2":
            sport_client = B2SportClient()
        else:
            raise ValueError("Unsupported robot type. Supported types are: go2, h1, b2.")
        sport_client.SetTimeout(10.0)
        sport_client.Init()
        return sport_client

    def _update_image_from_video_client(self):
        """Update Image from Robot's Built-in Camera"""
        code, data = self.video_client.GetImageSample()
        if code != 0:
            print("Failed to获取图像失败，错误代码:", code)
            return
        if isinstance(data, list):
            data = bytes(data)
        if len(data) == 0:
            return
        self.image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def _update_image_from_realsense(self):
        """Update Image from RealSense Camera"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            self.image = np.asanyarray(color_frame.get_data())

    def _yolo_image_post_process(self, results, original_image):
        """Process YOLO Detection Results"""
        detections = []
        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                if class_name == self.extracted_object:
                    # Convert normalized coordinates to pixel coordinates (xywhn to xyxy)
                    img_height, img_width = original_image.shape[:2]
                    x_center_n, y_center_n = box.xywhn[0][0], box.xywhn[0][1]
                    width_n, height_n = box.xywhn[0][2], box.xywhn[0][3]
                    x1 = int((x_center_n - width_n/2) * img_width)
                    y1 = int((y_center_n - height_n/2) * img_height)
                    x2 = int((x_center_n + width_n/2) * img_width)
                    y2 = int((y_center_n + height_n/2) * img_height)
                    
                    detections.append({
                        "object": class_name,
                        "confidence": float(box.conf),
                        "xyn": box.xywhn[0][:2].tolist(),
                        "whn": box.xywhn[0][2:].tolist(),
                        "xyxy": (x1, y1, x2, y2)  # Bounding box in pixel coordinates
                    })

        # Initialize history storage if not exists
        if not hasattr(self, 'history_confidence'):
                self.history_object = []
                self.history_confidence = []
                self.history_xyn = []
                self.history_whn = []
                self.history_xyxy = []
                self.last_best = None
                
        # Update detection history
        if detections:
            best = max(detections, key=lambda x: x["confidence"])
            self.last_best = best
            # Store detection results in a sliding window fashion
            self.history_object.append(best["object"])
            self.history_confidence.append(best["confidence"])
            self.history_xyn.append(best["xyn"])
            self.history_whn.append(best["whn"])
            self.history_xyxy.append(best["xyxy"])
            
            # Maintain fixed window size
            if len(self.history_object) > self.lengthen_filter:
                self.history_object.pop(0)
                self.history_confidence.pop(0)
                self.history_xyn.pop(0)
                self.history_whn.pop(0)
                self.history_xyxy.pop(0)
        else:
            # Handle case with no detections
            self.history_object.append("NULL")
            self.history_confidence.append(0.00)
            self.history_xyn.append(self.last_best["xyn"] if self.last_best else [0.00, 0.00])
            self.history_whn.append(self.last_best["whn"] if self.last_best else [0.00, 0.00])
            self.history_xyxy.append(self.last_best["xyxy"] if self.last_best else [0, 0, 0, 0])
            
            # Maintain fixed window size
            if len(self.history_object) > self.lengthen_filter:
                self.history_object.pop(0)
                self.history_confidence.pop(0)
                self.history_xyn.pop(0)
                self.history_whn.pop(0)
                self.history_xyxy.pop(0)
                
        # Calculate average values from history
        avg_confidence = np.mean(self.history_confidence)
        avg_xyn = np.mean(self.history_xyn, axis=0).tolist()
        avg_whn = np.mean(self.history_whn, axis=0).tolist()
        avg_xyxy = np.mean(self.history_xyxy, axis=0).tolist()
        avg_xyxy = [int(coord) for coord in avg_xyxy]

        # Find most common detected object
        most_common_object = max(set(self.history_object), key=self.history_object.count)
        
        # Handle case where most common object is NULL
        if most_common_object == "NULL":
            avg_confidence = 0.00
            avg_xyn = [0.00, 0.00]
            avg_whn = [0.00, 0.00]
            avg_xyxy = None

        # Update state with processed results
        self.state.update({
            "predicted_object": most_common_object,
            "confidence": [avg_confidence],
            "object_xyn": avg_xyn,
            "object_whn": avg_whn,
            "bounding_box": avg_xyxy
        })

    def _update_motion_control(self, state):
        """Update Motion Control Parameters Based on Detection Results"""
        input_data = {
            "mission_instruction_0": self.mission_instruction_0,
            "mission_instruction_1": self.mission_instruction_1,
            **state
        }
        prediction = self.motion_predictor.predict(input_data)
        self.state["mission_state_in"] = prediction["predicted_state"]
        self.state["search_state_in"] = prediction["search_state"]
        self.motion_vector = prediction["motion_vector"]

    def _control_robot(self):
        """Send Motion Commands to Robot"""
        if hasattr(self, 'motion_vector'):
            v_x, v_y, w_z = [float(val) for val in self.motion_vector]
            self.sport_client.Move(v_x, v_y, w_z)

    def _show_results(self, image):
        """Draw Detection Results and Information on Image"""
        # Draw bounding box if enabled and object detected
        if self.state["predicted_object"] != "NULL" and self.state["bounding_box"] is not None and self.show_max_result:
            x1, y1, x2, y2 = self.state["bounding_box"]
            confidence = self.state["confidence"][0]
            class_name = self.state["predicted_object"]
            object_cxy = self.state["object_xyn"]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green border
            
            # Draw label and confidence
            label = f"{class_name}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)  # Label background
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # Black text
            
            # Draw arrow from image center to object center if enabled
            if self.show_arrowed:
                image_center = (image.shape[1] // 2, image.shape[0] // 2)
                object_center = (int(object_cxy[0] * image.shape[1]), int(object_cxy[1] * image.shape[0]))
                cv2.arrowedLine(image, image_center, object_center, (255, 0, 255), 2)

                # Draw reference points
                cv2.circle(image, image_center, 10, (0, 255, 0), -1)  # Solid circle at center
                cv2.circle(image, object_center, 10, (0, 255, 255), 2)  # Hollow circle at object

        # Draw status information with black background
        texts = [
            f"Mission Instruction 1: {self.mission_instruction_1}",
            f"Mission Instruction 0: {self.mission_instruction_0}",
            f"Extracted Mission Object: {self.extracted_object}",
            f"Mission State In: {self.state['mission_state_in']}"
        ]
        if hasattr(self, 'motion_vector'):
            texts.append(f"Motion Vector: {self.motion_vector}")
        else:
            texts.append("Motion Vector: Not Available")

        y_positions = [30, 60, 90, 120, 150]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        font_color = (255, 255, 0)
        font_thickness = 2
        padding = 5  # Padding between text and background rectangle

        for text, y in zip(texts, y_positions):
            # Calculate text dimensions
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            # Draw black background rectangle
            x = 10
            rect_x = x - padding
            rect_y = y - text_height - padding
            rect_width = text_width + 2 * padding
            rect_height = text_height + baseline + 2 * padding
            cv2.rectangle(image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), -1)
            # Draw text
            cv2.putText(image, text, (x, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        return image

    def update_image(self):
        """Update Video Display in UI"""
        try:
            if hasattr(self.image_getter_thread, 'image_queue'):
                img = self.image_getter_thread.image_queue.get(timeout=1)
                img = self._show_results(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((1200, 800), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image=img)
                self.image_label.config(image=photo)
                self.image_label.image = photo
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Image update error: {e}")
        self.root.after(100, self.update_image)

    def start_threads(self):
        """Start All Worker Threads"""
        self.image_getter_thread.start()
        self.yolo_processing_thread.start()
        self.motion_control_thread.start()

    def stop_threads(self):
        """Stop All Worker Threads"""
        self.image_getter_thread.stop()
        self.yolo_processing_thread.stop()
        self.motion_control_thread.stop()
        if self.camera_type == "realsense":
            self.pipeline.stop()

    def run(self):
        """Main Run Method"""
        self.start_threads()
        self.root.after(100, self.update_ui_labels)
        self.root.after(100, self.update_freq_display)  # Start frequency update loop
        self.root.mainloop()
        self.stop_threads()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visual Language Motion Controller for Unitree Robots')
    
    # Model paths
    parser.add_argument('--yolo_model_dir', type=str, default="models/yolo-models/yolo11x.pt",
                      help='Path to YOLO model directory')
    parser.add_argument('--tokenizer_path', type=str, default="models/tokenizer_language2motion_n1000000",
                      help='Path to tokenizer')
    parser.add_argument('--object_extraction_model_path', type=str, 
                      default="models/model_object_extraction_n1000000_d64_h4_l2_f256_msl64_hold_success",
                      help='Path to object extraction model')
    parser.add_argument('--language2motion_model_path', type=str,
                      default="models/model_language2motion_n1000000_d128_h8_l4_f512_msl64_hold_success",
                      help='Path to language-to-motion model')
    
    # Hardware configuration
    parser.add_argument('--camera_type', type=str, default='realsense',
                      choices=['inner', 'realsense'], help='Camera type (inner or realsense)')
    parser.add_argument('--robot_type', type=str, default='go2',
                      choices=['go2', 'h1', 'b2'], help='Robot type (go2, h1, or b2)')
    
    # Display options
    parser.add_argument('--show_video', action='store_true', default=True,
                      help='Show video stream')
    parser.add_argument('--show_max_result', action='store_true', default=True,
                      help='Show detection results')
    parser.add_argument('--show_arrowed', action='store_true', default=False,
                      help='Show direction arrows')
    
    # Algorithm parameters
    parser.add_argument('--threshold', type=float, default=10.0,
                      help='Blur detection threshold')
    parser.add_argument('--lengthen_filter', type=int, default=1,
                      help='Number of historical detection results to keep')
    
    args = parser.parse_args()

    # Initialize and run controller
    controller = VisualLanguageController(
        yolo_model_dir=args.yolo_model_dir,
        tokenizer_path=args.tokenizer_path,
        object_extraction_model_path=args.object_extraction_model_path,
        language2motion_model_path=args.language2motion_model_path,
        camera_type=args.camera_type,
        robot_type=args.robot_type,
        show_video=args.show_video,
        show_max_result=args.show_max_result,
        show_arrowed=args.show_arrowed,
        blur_threshold=args.threshold,
        lengthen_filter=args.lengthen_filter
    )
    controller.run()
    print("Program terminated.")
