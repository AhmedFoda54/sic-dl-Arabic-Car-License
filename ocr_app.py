import streamlit as st
from PIL import Image
from utils import crop_LowerPart_Plate, detect_text_easyocr, detect_text_yolo


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
        st.spinner("Performing OCR on the cropped image...")
        easyocr_results = detect_text_easyocr(cropped_image)
        yolocr_results, nums, chars = detect_text_yolo(ocr_yolo_model, cropped_image)

        st.subheader("YOLO OCR")
        st.image(yolocr_results, caption="YOLO OCR", use_column_width=True)
        st.write(f"Detected Numbers: {nums} and detected Chars: {chars}")

        st.subheader("Easy OCR")
        st.spinner("Performing Easy OCR on the yolo image...")
        # Display OCR results
        if easyocr_results:
            st.write("Detected Text:")
            for text, confidence in easyocr_results:
                st.write(f"Text: {text}, Confidence: {confidence}")
        else:
            st.write("No text detected.")
    else:
        st.write("No car plate detected in the image.")
