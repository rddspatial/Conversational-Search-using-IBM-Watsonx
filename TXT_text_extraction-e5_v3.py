# After parsing the PDF for images and text the file is saved as TXT in the previous step
# In this code, that TXT file is processed and tokenized and chunks are created to ingest in elastic index
# This code does not create the embedding, it just chunks the text based on tokens
# For english and spanish we are using two embedding models here to tokenize the text in order to chunk 
# Tokenization is needed to decide the chunk size as multi-lingual-e5-small (embedding model) in elastic has a fixed chunk size (512)
# Take the TXT and save it to CSV

import os
import glob
import pandas as pd
from langdetect import detect, LangDetectException
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define base folder and input path
doc_folder = '' # input/source folder where a given TXT is saved after processing the PDF to interpret the images/diagrams and reterive texts
base_folder = '' # source path
input_folder = base_folder + "data_output_txt/"  # Folder containing your .txt files - modify the path as per your requirement
input_folder = f'./{doc_folder}/' # modify the path as per your requirement
output_folder = base_folder + f'data_output_txt2csv_v3/{doc_folder}/' #modify the path as per your need

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Collect all .txt paths
txt_path_list = glob.glob(os.path.join(input_folder, "*.txt"))
print(f"Found {len(txt_path_list)} text files")

# Define tokenizers for Spanish, English, and Arabic (if needed)
tokenizers = {
    'en': AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'),
    'es': AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
}

# Function to create text splitter based on language
def create_text_splitter(language_code):
    tokenizer = tokenizers.get(language_code)
    
    def length_function(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    
    text_splitter = RecursiveCharacterTextSplitter(
        length_function=length_function,
        chunk_size=250,
        chunk_overlap=100
    )
    
    return text_splitter

# Process each .txt file
for k, each_txt in enumerate(txt_path_list, start=1):
    print(f"{k}. Processing {each_txt}")
    text = []
    chunk_num = []
    title = []
    path = []
    
    try:
        with open(each_txt, 'r', encoding='utf-8') as file:
            full_text = file.read()
        
        # Detect language
        lang = detect(full_text)
        if lang not in ['en', 'es']:
            print(f"Skipping {each_txt}: Detected language {lang} is not supported.")
            continue
        
        # Create appropriate text splitter
        text_splitter = create_text_splitter(lang)
        chunks = text_splitter.split_text(full_text)
        
        # Determine the title
        txt_title = os.path.basename(each_txt).replace('.txt', '')

    except LangDetectException as e:
        print(f"Language detection failed for {each_txt}: {str(e)}")
        continue
    except Exception as error:
        print(f"An error occurred: {str(error)}")
        continue

    txt_path = os.path.relpath(each_txt, base_folder)

    for i, chunk in enumerate(chunks):
        title.append(txt_title)
        text.append(f"passage: {chunk.strip()}")
        chunk_num.append(i + 1)
        path.append(txt_path)

    # Convert lists to dataframe
    text_df = pd.DataFrame({
        'title': title,
        'path': path,
        'chunk num': chunk_num,
        'text': text
    })

    # Print DataFrame shape for verification
    print(text_df.shape)
    print(text_df.head())

    # Saving dataframe to CSV with the same name as the input .txt file
    output_file = os.path.join(output_folder, f'{txt_title}.csv')
    text_df.to_csv(output_file, index=False, header=True, encoding='utf_8_sig')

    print(f"Saved {output_file}")
