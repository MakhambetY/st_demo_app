import streamlit as st
import torch
import numpy as np
import cv2
import os
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


class ObjectDetection:
    def __init__(self, capture_index, save = False):
        """Initializes an ObjectDetection instance with a given camera index."""
        self.capture_index = capture_index
        self.save = save

        # model information
        self.model = YOLO("models/train-yolov8-n-100/weights/best.pt")
        # self.model = YOLO("yolov8n.pt")

        # visual information
        self.annotator = None
        self.start_time = 0
        self.end_time = 0

        # device information
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def predict(self, im0):
        """Run prediction using a YOLO model for the input image `im0`."""
        results = self.model(im0, conf=0.4)
        return results

    def display_fps(self, im0):
        """Displays the FPS on an image `im0` by calculating and overlaying as white text on a black rectangle."""
        self.end_time = time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(im0, (20 - gap, 70 - text_size[1] - gap), (20 + text_size[0] + gap, 70 + gap), (255, 255, 255), -1)
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        """Plots bounding boxes on an image given detection results; returns annotated image and class IDs."""
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            self.annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
        return im0, class_ids

    def display_total(self, results, im0):
        classes = results[0].boxes.cls.cpu().tolist()
        class_names = results[0].names
        total_kzt = 0

        # Dictionary to keep track of sign banknotes
        sign_banknotes = set()

        for cls in classes:
            class_name = class_names[int(cls)]

            # Check if it's a KZT banknote
            if cls <= 5:
                total_kzt += float(class_name.split('_')[0])
            # If it's not a KZT banknote, check if it's a sign
            else:
                if cls - 6 not in classes:
                    sign_banknotes.add(class_name)

        for sign in sign_banknotes:
            kzt_equivalent = int(sign.split('_')[0])
            total_kzt += kzt_equivalent

        print("Total KZT:", total_kzt)
        # Display total KZT
        kzt_text = f'Total KZT: {total_kzt}'
        kzt_text_size = cv2.getTextSize(kzt_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        kzt_gap = 10
        cv2.rectangle(im0, (20 - kzt_gap, 140 - kzt_text_size[1] - kzt_gap), (20 + kzt_text_size[0] + kzt_gap, 140 + kzt_gap), (255, 255, 255), -1)
        cv2.putText(im0, kzt_text, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def __call__(self):
        """Executes object detection on video frames from a specified camera index, plotting bounding boxes and returning modified frames."""
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Get video properties
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Unable to read video frame.")
        H, W, _ = frame.shape
        video_path_out = 'results/video_out_total.mp4'
        # Initialize output video writer
        out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
        if not out.isOpened():
            raise IOError("Unable to create video writer.")

        frame_count = 0
        while cap.isOpened():
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            self.display_fps(im0)
            self.display_total(results, im0)
            # Assuming you have an image loaded as 'frame'
            if self.save is True:
                # Write processed frame to output video
                out.write(im0)
            # Get current frame dimensions
            height, width = im0.shape[:2]

            # Reduce the size by half twice (adjust factor as needed)
            new_width = int(width / 2)
            new_height = int(height / 2)

            # Resize the frame (optional, for better quality)
            im0 = cv2.resize(im0, (new_width, new_height), interpolation=cv2.INTER_AREA)
            cv2.imshow('YOLOv8 Detection', im0)

            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()


def main():
    VIDEOS_DIR = "source/"
    st.title("Banknote Detection")

    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("**Select an option ** ")
    st.sidebar.write("")

    activities = [
        "Video(Video detection)", "Camera(live detection)"]
    choice = st.sidebar.selectbox("select an option", activities)

    if choice == "Video(Video detection)":
        video_file = st.file_uploader(
            "Upload Video", type=['avi', 'mp4', 'mov'])
        if video_file:
            with open(video_file, "wb") as f:
                f.write(video_file.getbuffer())

            video_path_out = 'results/video_out_total.mp4'
            if os.path.exists(video_file):
                cap = cv2.VideoCapture(video_file)
                if not cap.isOpened():
                    raise IOError("Unable to open video file.")

                # Get video properties
                ret, frame = cap.read()
                if not ret:
                    raise ValueError("Unable to read video frame.")
                H, W, _ = frame.shape
                out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'h264'), int(cap.get(cv2.CAP_PROP_FPS)),
                                      (W, H))
                if not out.isOpened():
                    raise IOError("Unable to create video writer.")

                model = YOLO(f'models/train-yolov8-n-100/weights/best.pt')

                while cap.isOpened():
                    success, frame = cap.read()

                    if success:
                        results = model.predict(frame)
                        annotated_frame = results[0].plot()
                        out.write(annotated_frame)

                    else:
                        break

                # Release resources
                cap.release()
                out.release()
                # cv2.destroyAllWindows()
                with open(video_path_out, "rb") as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)
    elif choice == "Camera(live detection)":
        st.title("Webcam Live Feed")
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        model = YOLO(f'models/train-yolov8-n-100/weights/best.pt')

        while run:
            _, frame = camera.read()

            results = model.predict(frame)
            annotated_frame = results[0].plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(annotated_frame_rgb)


if __name__ == "__main__":
    main()
