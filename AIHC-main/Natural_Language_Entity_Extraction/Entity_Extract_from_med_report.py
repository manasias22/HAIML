# !pip install pandas spacy
# !python -m spacy download en_core_web_sm
# from google.colab import files
import pandas as pd
import spacy
df = pd.read_csv('C:\\Users\\ASUS\\Sem 7 B.E\\AIHC\\Natural_Language_Entity_Extraction\\final_cbc_diagnoses_dataset_with_labels - final_cbc_diagnoses_dataset_with_labels.csv')
filename = 'final_cbc_diagnoses_dataset_with_labels.csv' 
df = pd.read_csv(filename)
nlp = spacy.load('en_core_web_sm')
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
df['entities'] = df['long_title'].apply(lambda x: extract_entities(str(x)))
output_filename = 'final_cbc_with_entities.csv'
df.to_csv(output_filename, index=False)
files.download(output_filename)

# if error does occur :( IGY
    # pip install pandas
    # pip install --upgrade pandas pyarrow
    # pip install numpy<2

