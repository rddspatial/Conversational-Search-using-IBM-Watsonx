# run this code in docling_env conda/virtual env
# conda activate docling_env
# this will parse the PDF and reteive texts and interpret images/diagrams using Llama 3.2 vision and docling from IBM 
# Take the PDF and save it to TXT

import os
import PyPDF2
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument, PDFTextExtractionNotAllowed, PDFEncryptionError
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTImage
import requests as rq
import base64
import sys
import pandas as pd
from docling.document_converter import DocumentConverter  # Import docling package

# Watsonx.ai API setup
src = './credential.csv'
df = pd.read_csv(src)
CLOUD_API_KEY = df['CLOUD_API_KEY'].iloc[0]

PROJECT_ID = '' # watsonx project id
URL = 'https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2023-05-29'
IAM_URI = 'https://iam.cloud.ibm.com/identity/token'
IAM_DATA = f'grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={CLOUD_API_KEY}'
IAM_HEADERS = {'Content-Type': 'application/x-www-form-urlencoded'}

res = rq.post(IAM_URI, data=IAM_DATA, headers=IAM_HEADERS)
TOKEN = res.json()['access_token']

HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {TOKEN}'
}

# Watsonx.ai LLaMA 3.2 Vision model function
def interpret_image_with_llama(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"
    
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Explain the image with each and every detail including image description and its components, their dimensions, angles, orientation if any, and also the labels or legends in the image. Generate your response in the original language as that of in the image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_string}"
                        }
                    }
                ]
            }
        ],
        "project_id": PROJECT_ID,
        "model_id": "meta-llama/llama-3-2-90b-vision-instruct",
        "decoding_method": "greedy",
        "repetition_penalty": 1,
        "max_tokens": 900
    }

    response = rq.post(url, headers=HEADERS, json=body)

    if response.status_code != 200:
        raise Exception(f"Error in image processing: {response.text}")

    output = response.json()['choices'][0]['message']['content']
    return output

# Detect embedded images using pdfminer and handle encrypted PDFs using docling
def extract_images_and_text(pdf_path):
    images_info = []
    text_content = []

    try:
        with open(pdf_path, 'rb') as fp:
            parser = PDFParser(fp)
            document = PDFDocument(parser)

            if not document.is_extractable:
                raise PDFTextExtractionNotAllowed

            rsrcmgr = PDFResourceManager()
            laparams = LAParams()
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)

            # Extract text and check for image objects on each page
            for page_num, page in enumerate(PDFPage.create_pages(document)):
                interpreter.process_page(page)
                layout = device.get_result()
                
                page_text = ''
                page_images = []

                for element in layout:
                    if hasattr(element, "get_text"):
                        page_text += element.get_text()
                    if isinstance(element, list):
                        for obj in element:
                            if obj.__class__.__name__ == "LTImage":
                                page_images.append(f"Image found on page {page_num + 1}")

                text_content.append(page_text if page_text else None)
                images_info.append(page_images if page_images else None)

    except (PDFTextExtractionNotAllowed, PDFEncryptionError):
        # Use docling if pdfminer cannot extract text
        print("Using docling for extraction due to encryption or extraction restriction.")
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        markdown_text = result.document.export_to_markdown()
        
        # Convert markdown to plain text for compatibility
        plain_text = markdown_text.replace("## ", "\n").replace("### ", "\n").replace("* ", "").replace("`", "")
        text_content.append(plain_text)
        images_info.append(None)  # No image extraction via docling

    return text_content, images_info, len(text_content)

# Updated function to handle the text-image extraction and processing with LLaMA
def process_pdf_with_images(pdf_path, output_folder, input_filename):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract text and images from PDF
    text_content, images_info, total_pages = extract_images_and_text(pdf_path)

    # Handle cases where no text or images were extracted
    if text_content is None and images_info is None:
        print("No text or images extracted. Exiting.")
        return

    print(f"Total Pages in PDF: {total_pages}")

    final_text = []

    for i in range(total_pages):
        current_page = i + 1

        # Show progress for each page
        sys.stdout.write(f"Processing page {current_page}/{total_pages}...\n")
        sys.stdout.flush()

        # If images are present on the page, interpret them with LLaMA
        if images_info[i]:
            image_explanation = interpret_image_with_llama(images_info[i][0])  # Modify to process each image if multiple
            concatenated_text = text_content[i] + "\n\n" + image_explanation if text_content[i] else image_explanation
        else:
            concatenated_text = text_content[i] if text_content[i] else ""

        final_text.append(concatenated_text)

    # Save final text to .txt
    final_output = input_filename + '.txt'
    with open(os.path.join(output_folder, final_output), 'w') as f:
        f.write("\n\n".join(final_text))

    print(f"\nProcessing complete. Output saved to {output_folder}.")

# Example usage
input_filename = '' # chapter name (without .PDF)
pdf_path = '' + input_filename + '.pdf' # provide source path
output_folder = '' + input_filename + '/' # provide target path
process_pdf_with_images(pdf_path, output_folder, input_filename)
