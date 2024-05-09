
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import os


def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)



def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    # Get the file extension of the input video
                    file_extension = source_video.name.split('.')[-1].lower()

                    # Create a temporary file to write the input video content
                    with tempfile.NamedTemporaryFile(delete=False) as tfile:
                        tfile.write(source_video.read())

                        # Open the input video file
                        vid_cap = cv2.VideoCapture(tfile.name)

                        # Get the original width and height of the input video frames
                        width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        # Define the output video file name with the same extension as the input video
                        output_file_name = f"output.{file_extension}"

                        # Define the codec based on the file extension
                        codec = 'mp4v' if file_extension == 'mp4' else 'XVID'

                        # Define the codec and create a VideoWriter object to write the output video
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        out = cv2.VideoWriter(output_file_name, fourcc, 30.0, (width, height))

                        # Initialize an empty Streamlit frame to display the output video
                        st_frame = st.empty()

                        # Process each frame of the input video
                        while vid_cap.isOpened():
                            success, image = vid_cap.read()
                            if success:
                                # Resize the image to a standard size
                                image_resized = cv2.resize(image, (720, int(720 * (height / width))))

                                # Predict the objects in the image using YOLOv8 model
                                res = model.predict(image_resized, conf=conf)

                                # Plot the detected objects on the video frame
                                res_plotted = res[0].plot()
                                st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)

                                # Write the processed frame to the output video file
                                out.write(res_plotted)

                            else:
                                break

                        # Release the VideoCapture and VideoWriter objects
                        vid_cap.release()
                        out.release()

                    # Delete the temporary file
                    os.unlink(tfile.name)

                except Exception as e:
                    st.error(f"Error processing video: {e}")

    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    # Get the file extension of the input video
                    file_extension = source_video.name.split('.')[-1].lower()

                    # Create a temporary file to write the input video content
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(source_video.read())
                    tfile.close()

                    # Open the input video file
                    vid_cap = cv2.VideoCapture(tfile.name)

                    # Get the original width and height of the input video frames
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # Define the output video file name with the same extension as the input video
                    output_file_name = f"output.{file_extension}"

                    # Define the codec and create a VideoWriter object to write the output video
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if file_extension == 'mp4' else cv2.VideoWriter_fourcc(*'MOV')
                    out = cv2.VideoWriter(output_file_name, fourcc, 30.0, (width, height))

                    # Initialize an empty Streamlit frame to display the output video
                    st_frame = st.empty()

                    # Process each frame of the input video
                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            # Resize the image to a standard size
                            image_resized = cv2.resize(image, (720, int(720 * (height / width))))

                            # Predict the objects in the image using YOLOv8 model
                            res = model.predict(image_resized, conf=conf)

                            # Plot the detected objects on the video frame
                            res_plotted = res[0].plot()
                            st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)

                            # Write the processed frame to the output video file
                            out.write(res_plotted)

                        else:
                            break

                    # Release the VideoCapture and VideoWriter objects
                    vid_cap.release()
                    out.release()

                except Exception as e:
                    st.error(f"Error processing video: {e}")

                    # Delete the temporary file
                    os.unlink(tfile.name)


def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")






# DuplicateWidgetID: There are multiple identical st.file_uploader widgets with the same generated key.

# When a widget is created, it's assigned an internal key based on its structure. Multiple widgets with an identical structure will result in the same internal key, which causes this error.

# To fix this error, please pass a unique key argument to st.file_uploader.

# Traceback:
# File "C:\Users\Пользователь\Downloads\Telegram Desktop\diplomaproject\diplomaproject\new_app.py", line 73, in <module>
#     infer_uploaded_video(confidence, model)
# File "C:\Users\Пользователь\Downloads\Telegram Desktop\diplomaproject\diplomaproject\utils.py", line 177, in infer_uploaded_video
#     source_video = st.sidebar.file_uploader(