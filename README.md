# 🖼️ OCR Web Application using Qwen2-VL and Streamlit

This project demonstrates an advanced Optical Character Recognition (OCR) web application designed to extract multilingual text (**Swedish and English**) from images using Hugging Face's **Qwen2-VL vision-language transformer model**. The app is built using **Streamlit**, providing a user-friendly interface for uploading images, viewing extracted text, and performing keyword searches.

---

## 📌 Project Description

The OCR application involves:

- **OCR Extraction**: Uses Hugging Face's **Qwen2-VL-2B-Instruct** model to extract text accurately from images containing Swedish and English content.
- **Streamlit Web Interface**: A user-friendly interface to upload images, instantly view the extracted text, and perform keyword searches with highlighted results.
- **Simple Deployment**: Can easily be deployed to platforms like **Streamlit Community Cloud** or **Hugging Face Spaces** for easy public access.

---

## ✨ Features

- **Multilingual OCR**: Accurately extracts both Swedish and English texts from uploaded images.
- **Keyword Search & Highlighting**: Easily search for and highlight keywords within the extracted text.
- **User-Friendly UI**: Built with Streamlit for simplicity and intuitive interaction.

---

## ⚙️ Technical Requirements

- Python 3.8 or higher
- Dependencies specified in `requirements.txt`

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/AbdullahDhrubo/Multilingual-OCR.git
    cd Multilingual_OCR
    ```

2. Set up the virtual environment:

    ```bash
    python -m venv .venv
    source .venv\Scripts\activate # For Windows
    source .venv/bin/activate  # For Mac
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Once the installation is complete, you can start the Streamlit app:

```bash
streamlit run app.py
```
