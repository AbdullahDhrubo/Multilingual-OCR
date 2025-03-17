import streamlit as st
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info


# Load the OCR model with caching to optimize performance
@st.cache_resource
def load_model():
    model = (
        Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .cpu()
        .eval()
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return model, processor


# Convert the uploaded image into RGB format for consistency
def preprocess_image(image):
    return image.convert("RGB")


# Perform OCR extraction on the image using the pre-loaded model
def extract_text_from_image(image, model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": "Extract all text from this image, including both Swedish and English.",
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cpu")

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    trimmed_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    extracted_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return extracted_text


# Highlight searched keywords within the extracted text
def highlight_text(text, keyword):
    return text.replace(keyword, f"<mark>{keyword}</mark>")


# Streamlit Application UI
st.set_page_config(page_title="OCR App with Byaldi & Qwen2-VL", layout="centered")
st.title("🖼️ Multilingual OCR Application")
st.markdown(
    "Upload an image to instantly extract and search for multilingual (Swedish & English) text."
)

# Image upload interface with clearer instructions
uploaded_file = st.file_uploader(
    "Upload an image (Supported formats: JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Uploaded Image", use_container_width=True)

    # Inform user about the processing status
    with st.spinner("Extracting text from image..."):
        model, processor = load_model()
        processed_image = preprocess_image(input_image)
        extracted_text = extract_text_from_image(processed_image, model, processor)

    # Convert extracted text to a clean, single string
    extracted_text_str = " ".join(extracted_text).strip()

    # Display extracted text with clear UI separation
    st.subheader("📝 Extracted Text:")
    st.info(extracted_text_str)

    # Search functionality with better usability
    st.subheader("🔍 Keyword Search")
    search_query = st.text_input("Enter keywords to highlight in the extracted text:")

    if search_query:
        if search_query.lower() in extracted_text_str.lower():
            highlighted_result = highlight_text(extracted_text_str, search_query)
            st.markdown(
                f"<div style='padding:10px;border-radius:5px;background-color:#e0f7fa;'>{highlighted_result}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("No matches found for your query. Please try another keyword.")
