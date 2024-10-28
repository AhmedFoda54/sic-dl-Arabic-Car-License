import streamlit as st
from PIL import Image
from utils import crop_LowerPart_Plate, detect_text_easyocr, detect_text_yolo, process_video_with_plate_detection

yolo_model = 'models/yolo11m_car_plate_trained.pt'
ocr_yolo_model = 'models/yolo11m_car_plate_ocr.pt'

# Streamlit app interface
st.title("Car Plate Detection and OCR")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily for processing
    image_path = f"/tmp/{uploaded_file.name}"
    image = Image.open(uploaded_file)
    image.save(image_path)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Crop the plate from the image
    st.write("Cropping the car plate...")
    cropped_image = crop_LowerPart_Plate(yolo_model, image_path)

    if cropped_image:
        st.image(cropped_image, caption="Cropped Car Plate", use_column_width=True)

        # Perform OCR on the cropped image
        with st.spinner("Performing OCR on the cropped image..."):
            easyocr_image, easyocr_text = detect_text_easyocr(cropped_image)
            yolo_results, nums, chars = detect_text_yolo(ocr_yolo_model, cropped_image)

        st.subheader("YOLO OCR")
        st.image(yolo_results, caption="YOLO OCR", use_column_width=True)
        st.write(f"Detected Numbers: {nums} and detected Chars: {chars}")

        st.subheader("Easy OCR")
        st.image(easyocr_image, caption="EasyOCR", use_column_width=True)
        if easyocr_text:
            st.write("Detected Text:")
            for text, confidence in easyocr_text:
                st.write(f"Text: {text}, Confidence: {confidence}")
        else:
            st.write("No text detected.")
    else:
        st.write("No car plate detected in the image.")

# Streamlit application for video uploading and processing
st.title("Car Plate Detection from Video")

# Upload video file
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Process the video if a file is uploaded
if uploaded_video is not None:
    st.video(uploaded_video)

    with st.spinner("Processing video..."):
        # Save processed video to /tmp directory and display it
        processed_video_path = process_video_with_plate_detection(uploaded_video, yolo_model, ocr_yolo_model)
        
        if processed_video_path:
            # Display the processed video
            st.video(processed_video_path)
        else:
            st.write("There was an issue processing the video.")
