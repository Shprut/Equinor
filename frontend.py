import streamlit as st
import transformers
import torch
import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
import re
import PyPDF4
from typing import List
import os
import docx2txt
from transformers import AutoTokenizer, BertForQuestionAnswering, AutoModelForQuestionAnswering, AutoModel

st.set_page_config(page_title="Equinor Data Catalog", page_icon=":mag:")


def dat_sor():
    
    

            # Load model
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

        # Function to get the answer to the question
        def answer_(text, question):
            inputs = tokenizer(question, text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                outputmodel = model(input_ids=input_ids, attention_mask=attention_mask)
                startscore, endscore = outputmodel[0], outputmodel[1]

            answer_start = torch.argmax(startscore)
            answer_end = torch.argmax(endscore) + 1

            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
            return answer

        def main():

            # Get a list of uploaded files
            files = get_uploaded_files()

            # Create the search bar
          

            # Show live suggestions based on the search term as the user types
            suggestions = [file for file in files if file.endswith(('.txt', '.pdf'))]

            # Create a multiselect widget to select files based on suggestions
            selected_files = st.multiselect("Search:", suggestions)

            # Display the selected files and their contents
            if selected_files:
                st.write("Selected files:")
                for file in selected_files:
                    st.write(file)
                    content = read_file(file)
                    st.text(content)

            # User inputs
            
            files = st.file_uploader("Choose a file", accept_multiple_files=True, type=['txt','docx','pdf'])
            question = st.text_input('Enter a question:')
            

            # Process the uploaded files in a loop and checks for document type to process each document for the document type
            if files:
                st.write('Answer:')
                for file in files:
                    file_extension = file.name.split('.')[-1]
                    text = ''
                    if file_extension == 'txt':
                        content = file.read().decode('utf-8')
                        text += content
                    elif file_extension == 'docx':
                        content = docx2txt.process(file)
                        text += content
                    elif file_extension == 'pdf':
                        pdf_reader = PyPDF4.PdfFileReader(file)
                        text = ""
                        for page_num in range(pdf_reader.getNumPages()):
                            page = pdf_reader.getPage(page_num)
                            text += page.extractText()

                    # Get the answer for the current file
                    answer = answer_(text, question)
                    # Display the answer
                    
                    st.write('\n')
                    if answer:
                        st.write(f'Answer for {file.name}: {answer}')
                        
                    else:
                        st.write(f'No answer found for {file.name}.')


        if __name__ == '__main__':
            main()
            st.write("")

# Define a function to get a list of uploaded files in the current directory
def get_uploaded_files():
    files = []
    for item in os.listdir("."):
        if os.path.isfile(item):
            files.append(item)
    return files

# Define a function to read the contents of a file
def read_file(file):
    with open(file, "r") as f:
        content = f.read()
    return content


# Display the image
st.image("equinor.png", width=200)

dat_sor()