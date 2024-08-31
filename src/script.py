import os
import sys
import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import shutil
from tqdm import tqdm
from utils import preprocess_text, extract_text_from_pdf, categorize_resume

def main(resume_directory):
    # Get the base directory (parent of src)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Loading the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=24)
    model_path = os.path.join(base_dir, 'model', 'resume_categorization_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load the tokenizer and label encoder
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    le = LabelEncoder()
    csv_path = os.path.join(base_dir, 'dataset', 'Resume.csv')
    le.classes_ = pd.read_csv(csv_path)['Category'].unique()

    # Create a list to store categorization results
    results = []

    # Ensure output directory exists
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Process each PDF in the directory
    for filename in tqdm(os.listdir(resume_directory)):
        if filename.endswith('.pdf'):
            file_path = os.path.join(resume_directory, filename)
            
            try:
                # Extract text from PDF
                raw_text = extract_text_from_pdf(file_path)
                
                # Preprocess the text
                processed_text = preprocess_text(raw_text)
                
                # Categorize the resume
                category = categorize_resume(processed_text, model, tokenizer, le, device)
                
                # Create category folder if it doesn't exist
                category_folder = os.path.join(output_dir, category)
                os.makedirs(category_folder, exist_ok=True)
                
                # Copy the file to the category folder (instead of moving)
                new_file_path = os.path.join(category_folder, filename)
                shutil.copy2(file_path, new_file_path)
                
                # Add result to the list
                results.append({'filename': filename, 'category': category})
            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Create a DataFrame from the results and save it as CSV
    df = pd.DataFrame(results)
    csv_output_path = os.path.join(output_dir, 'categorized_resumes.csv')
    df.to_csv(csv_output_path, index=False)

    print(f"Categorization complete. Results saved to '{csv_output_path}'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <resume_directory>")
        sys.exit(1)
    
    resume_directory = sys.argv[1]
    main(resume_directory)
