In this project we will create a pipeline to develop a conversational search AI using IBM watsonx, watsonx discovery and Watson Assistant (Watsonx Orchestrate).
The initial guideline shows you how to create the index in the knowledge base and process and ingest the data.

For this we will use
watsonx.ai to call Llama 3.2 vision model to parse images/diagrams in PDF
watsonx discovery to create the index in the knowledge base
watsonx orchestrate (watson assistant) to create the chat interface

Sequence of workflow:

1. config-e5_embedding.ipynb (only once to download the model and create the pipeline and index)

2. PDF_text_image_parser_v4.py.py (this may need to install some specific dependency for the first time such as brew install poppler, pip install docling)

docling works on python version >=3.11
docling can extract text from protected/encyrpted pdf. It is an OS text extraction package from IBM.
(make sure you use docling_env virtual/conda env which needs python>=3.10. 
conda activate docling_env
Currently I am using python 3.11 in docling_env conda virtual env. which basically first runs Llama 3.2 vision model to parse the images, if it somehow fails due to limitation on the PDF file, 
code will call docling DocumentConverter class and associated functions to parse the page/image

Initially parsing the PDF we are using opensource pdfminer which can detect which page has text and which has image. 
If it is image code calls LLama 3.2 vision model. 
However, sometimes we can encounter an error PDFEncryptionError: Unsupported revision indicates that the PDF file is encrypted with a security protocol that pdfminer does not support. 
To handle this, we can modify the code to skip pdfminer processing and go directly to the docling package for such encrypted files.
)


3. TXT_text_extraction-e5_v3.py
4. ingest-e5_embedding.ipynb
5. query-e5.ipynb
