import os
import streamlit as st
import pandas as pd
from huggingface_hub import HfApi
from io import StringIO

# Hugging Face credentials
#HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
HUGGINGFACE_TOKEN =
DATASET_REPO = 'your-username/your-dataset'
BASE_CSV_NAME = 'stories.csv'
MAX_FILE_SIZE_MB = 512

def main():
    st.title("Sinhala Story Submission")
    st.write("Submit your Sinhala story to be included in the dataset.")

    # Input form
    story = st.text_area("Enter your story in Sinhala:", height=200)

    if st.button("Submit"):
        if story.strip() == "":
            st.error("Please enter a story.")
        else:
            # Add story to dataset
            success = add_story_to_dataset(story)
            if success:
                st.success("Thank you for your submission!")
            else:
                st.error("Failed to submit your story. Please try again.")

def add_story_to_dataset(story):
    try:
        # Load existing dataset from Hugging Face
        api = HfApi()
        
        # Determine which file to use
        current_file = determine_current_file(api)
        
        # Download the current file
        dataset_info = api.dataset_info(DATASET_REPO)
        data_url = next(f.rfilename for f in dataset_info.siblings if f.rfilename == current_file)
        
        data = api.download_url(data_url, use_auth_token=HUGGINGFACE_TOKEN)
        data = pd.read_csv(StringIO(data.decode('utf-8')))
        
        # Append new story
        new_entry = pd.DataFrame({"story": [story]})
        updated_data = pd.concat([data, new_entry], ignore_index=True)
        
        # Convert updated data to CSV
        csv_buffer = StringIO()
        updated_data.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_content = csv_buffer.getvalue()
        
        # Upload updated CSV to Hugging Face
        api.upload_file(
            path_or_fileobj=csv_content,
            path_in_repo=current_file,
            repo_id=DATASET_REPO,
            repo_type='dataset',
            commit_message=f'Add new story to {current_file}'
        )
        return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

def determine_current_file(api):
    # Get the list of files in the dataset
    dataset_info = api.dataset_info(DATASET_REPO)
    files = [f.rfilename for f in dataset_info.siblings if f.rfilename.startswith(BASE_CSV_NAME)]
    
    # Determine the current file based on size
    for file in sorted(files, key=lambda x: int(x.lstrip('stories').rstrip('.csv') or '0')):
        data_url = next(f.rfilename for f in dataset_info.siblings if f.rfilename == file)
        data = api.download_url(data_url, use_auth_token=HUGGINGFACE_TOKEN)
        size_mb = len(data) / (1024 * 1024)
        if size_mb < MAX_FILE_SIZE_MB:
            return file
    
    # If all existing files exceed the size, create a new file
    new_file_index = len(files) + 1
    return f'stories{new_file_index}.csv'

if __name__ == "__main__":
    main()
