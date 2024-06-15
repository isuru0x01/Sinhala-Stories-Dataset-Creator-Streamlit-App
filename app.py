import os
import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, login, hf_hub_download, HfFolder, Repository
from io import StringIO
import re
from datasets import load_dataset, concatenate_datasets
from datasets import DatasetDict
from datasets import Dataset

# Hugging Face credentials
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
#HUGGINGFACE_TOKEN = "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"
DATASET_REPO = 'Isuru0x01/sinhala_stories'
BASE_CSV_NAME = 'stories.csv'
MAX_FILE_SIZE_MB = 512

# Streamlit app
st.title('Story Submission')

# Text input for user to submit a story
story = st.text_area('Enter your story here', height=300)

# Button to submit the story
if st.button('Submit'):
    
    progress = st.progress(0)
      
    new_entry = pd.DataFrame({"story": [story]})
    new_entry.to_json(f"new_entry.jsonl", orient="records", lines=True)
    progress.progress(0.25)  # Update progress bar
    
    st.write(new_entry)
    new_entry_dataset = load_dataset("json", data_files="new_entry.jsonl", split="train")
    progress.progress(0.5)  # Update progress bar
    
    st.write(new_entry_dataset)
    
    # Download the existing dataset from the Hugging Face hub
    existing_dataset = load_dataset(f"{DATASET_REPO}", split="train")
    progress.progress(0.75)  # Update progress bar
    
    # Append the new entry to the existing dataset
    updated_dataset = concatenate_datasets([existing_dataset, new_entry_dataset])
    
    # Push the updated dataset to the Hugging Face hub
    updated_dataset.push_to_hub(f"{DATASET_REPO}")
    progress.progress(1.0)  # Update progress bar