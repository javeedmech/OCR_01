import cv2
import tkinter as tk
from tkinter import Button, Label
import numpy as np
from PIL import Image, ImageTk
from paddleocr import PaddleOCR, draw_ocr


# Specify the model paths
ocr = PaddleOCR(
    det_model_dir='./inference/en_PP-OCRv3_det_infer',
    rec_model_dir='./inference/en_PP-OCRv3_rec_infer',
    lang='en'
)




# Initialize the OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Use 'ch' for Chinese, 'en' for English

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Constants for cropping the image
CROP_WIDTH = 400
CROP_HEIGHT = 300


def crop_center(frame, crop_width, crop_height):
    """
    Crops the frame from the center with the given width and height.

    Args:
        frame: The input image frame to crop.
        crop_width: The width of the cropped area.
        crop_height: The height of the cropped area.

    Returns:
        The cropped frame.
    """
    h, w, _ = frame.shape
    start_x = (w - crop_width) // 2
    start_y = (h - crop_height) // 2
    return frame[start_y:start_y + crop_height, start_x:start_x + crop_width]


def update_video_stream():
    """
    Updates the video stream continuously and displays it on the Tkinter GUI.
    """
    ret, frame = cap.read()
    if ret:
        cropped_frame = crop_center(frame, CROP_WIDTH, CROP_HEIGHT)
        rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the video label with the new frame
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

    # Schedule the next frame update
    video_label.after(10, update_video_stream)


def capture_image():
    """
    Captures the current frame from the video stream, crops it, and performs OCR.
    Saves the cropped image and displays the result.
    """
    ret, frame = cap.read()
    if ret:
        cropped_frame = crop_center(frame, CROP_WIDTH, CROP_HEIGHT)
        filename = "captured_image.jpg"
        
        cv2.imwrite(filename, cropped_frame)
        print(f"Image captured and saved as '{filename}'")

        # Run OCR on the captured image
        run_ocr(filename)



def run_ocr(image_path):
    """
    Runs OCR on the captured image and processes the results.

    Args:
        image_path: Path to the image to perform OCR on.
    """
    try:
        image_path=cv2.imread(image_path)



        text_result = ocr.ocr(image_path, cls=True)

        # Check if OCR result is valid
        if text_result is None or not text_result or text_result[0] is None or len(text_result[0]) == 0:
            display_no_text_detected()
        else:
            display_text_detected(image_path, text_result)

    except Exception as e:
        print(f"Error occurred during OCR: {e}")


def display_no_text_detected():
    """
    Displays a message on the GUI when no text is detected in the image.
    """
    label = tk.Label(root, text="No Text Detected in Image", fg="red", font=("Helvetica", 16))
    label.pack()


def display_text_detected(image_path, text_result):
    """
    Displays the image with detected text and saves the OCR results to a file.

    Args:
        image_path: Path to the image that has the detected text.
        text_result: The result from the OCR process.
    """
    for row in text_result[0]:
        bbox = [[int(r[0]), int(r[1])] for r in row[0]]
        cv2.putText(image_path, row[1][0], bbox[0], cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 255), 2)
        cv2.polylines(image_path, [np.array(bbox)], True, (255, 0, 255), 3)

    rgb_frame_o = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    img_o = Image.fromarray(rgb_frame_o)
    imgtk_o = ImageTk.PhotoImage(image=img_o)

    # Display the processed image with OCR text
    captured_label_o.imgtk = imgtk_o
    captured_label_o.config(image=imgtk_o)

    # Save OCR results to a file
    save_ocr_results_to_file(text_result)


def save_ocr_results_to_file(text_result):
    """
    Saves the OCR results (detected text) to a text file.

    Args:
        text_result: The result from the OCR process.
    """
    with open("ocr_output.txt", "a") as file:
        for data in text_result[0]:
            file.write(str(data[1][0]) + "\n")


# Create the main application window
root = tk.Tk()
root.title("JVD Text Reader")

# Create a label to display the video stream
video_label = Label(root)
video_label.pack()

# Create a button to capture an image
capture_button = Button(root, text="Capture Image", command=capture_image)
capture_button.pack()

# Create a label to display the captured image with OCR results
captured_label_o = Label(root, text="Captured Image with Text")
captured_label_o.pack()

# Start the video stream
update_video_stream()

# Run the Tkinter event loop
root.mainloop()

# Release the webcam when the application is closed
cap.release()
cv2.destroyAllWindows()