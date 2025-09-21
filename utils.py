import spacy
import nltk
import sys
import scispacy
import pandas as pd
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import re
import os
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel
import joblib  
import ast
import json
import streamlit as st

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

@st.cache_data
def preprocess_data(input_df):
    tqdm.pandas()

    # Load saved embedding PCA
    pca_model_path = '/ibex/project/c2205/AMR_dataset_peijun/data_before/data/trained_note_embedding_ipca_model.joblib'
    ipca = joblib.load(pca_model_path)

    # Load ClinicalBERT model and tokenizer
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Function to convert notes to embeddings
    def get_note_embedding_pca(note):
        if pd.isna(note):
            return None
        inputs = tokenizer(note, padding=True, max_length=512, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embedding_pca = ipca.transform(embedding.reshape(1, -1))
        return embedding_pca.squeeze()

    def name_extractor(text):
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        output = [tagged[i][0] for i in range(len(tagged)) if tagged[i][1] in ['NN', 'NNP', 'NNS', 'NNPS']]
        return ' '.join(output)

    def processor(sentence, minimum_length=3):
        stop_words = set(stopwords.words('english')) 
        lemmatizer = WordNetLemmatizer()
        sentence_lower = sentence.lower()
        tmp = [re.sub('[^A-Za-z]+', '', i) for i in word_tokenize(sentence_lower)]
        sentence_lower_nospecialchar = [i for i in tmp if len(i) > 0]
        sentence_lower_nospecialchar_nostopword = [w for w in sentence_lower_nospecialchar if w not in stop_words] 
        sentence_lower_nospecialchar_nostopword_lemmatized = [lemmatizer.lemmatize(i, pos='n') for i in sentence_lower_nospecialchar_nostopword] 
        sentence_lower_nospecialchar_nostopword_lemmatized_deduplicated = list(set(sentence_lower_nospecialchar_nostopword_lemmatized))
        sentence_lower_nospecialchar_nostopword_lemmatized_deduplicated_fileteredshorts = [i for i in sentence_lower_nospecialchar_nostopword_lemmatized_deduplicated if len(i) > minimum_length]
        return ' '.join(sentence_lower_nospecialchar_nostopword_lemmatized_deduplicated_fileteredshorts)

    def convert_and_preprocess(ner_str):
        try:
            entities = ast.literal_eval(ner_str)
            return list(set(entity.strip().rstrip(',') for entity in entities if entity.strip()))
        except (ValueError, SyntaxError):
            return []
    print('Creating note embedding and run pca')
    input_df['note_embedding_pca'] = input_df['additional_note'].progress_apply(get_note_embedding_pca)
    
    pca_columns = [f'note_embedding_pca_{i}' for i in range(ipca.n_components_)]
    pca_embeddings_input_df = pd.DataFrame(input_df['note_embedding_pca'].tolist(), columns=pca_columns)

    input_df = pd.concat([input_df.drop(columns=['note_embedding_pca']), pca_embeddings_input_df], axis=1)

    nlp_sci_lg = spacy.load("en_core_sci_lg")
    nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md")
    nlp_bionlp13cg = spacy.load("en_ner_bionlp13cg_md")

    ner_results = {
        'output_terms_nlp_sci_lg': [],
        'output_terms_nlp_bc5cdr': [],
        'output_terms_nlp_bionlp13cg': []
    }

    total_notes = len(input_df)
    for i, nt in enumerate(input_df['additional_note']):
        if isinstance(nt, str):
            input_sentence = processor(name_extractor(nt))
            
            doc_sci_lg = nlp_sci_lg(input_sentence)
            ner_results['output_terms_nlp_sci_lg'].append([ent.text for ent in doc_sci_lg.ents])
            
            doc_bc5cdr = nlp_bc5cdr(input_sentence)
            ner_results['output_terms_nlp_bc5cdr'].append([ent.text for ent in doc_bc5cdr.ents])
            
            doc_bionlp13cg = nlp_bionlp13cg(input_sentence)
            ner_results['output_terms_nlp_bionlp13cg'].append([ent.text for ent in doc_bionlp13cg.ents])
        else:
            ner_results['output_terms_nlp_sci_lg'].append(np.nan)
            ner_results['output_terms_nlp_bc5cdr'].append(np.nan)
            ner_results['output_terms_nlp_bionlp13cg'].append(np.nan)

        print(f"Progress: {100 * (i + 1) / total_notes:.2f}%")
        sys.stdout.flush()

    input_df['output_terms_nlp_sci_lg'] = ner_results['output_terms_nlp_sci_lg']
    input_df['output_terms_nlp_bc5cdr'] = ner_results['output_terms_nlp_bc5cdr']
    input_df['output_terms_nlp_bionlp13cg'] = ner_results['output_terms_nlp_bionlp13cg']

    input_df['all_ner'] = input_df.apply(lambda row: set(row['output_terms_nlp_sci_lg'] + 
                                                           row['output_terms_nlp_bc5cdr'] + 
                                                           row['output_terms_nlp_bionlp13cg']), axis=1)

    ipca_model_path = '/ibex/project/c2205/AMR_dataset_peijun/data_before/data/trained_ipca_model.joblib'
    ipca = joblib.load(ipca_model_path)

    filtered_entities_path = '/ibex/project/c2205/AMR_dataset_peijun/integrate/filtered_entities.csv'
    filtered_entities_df = pd.read_csv(filtered_entities_path, header=0)
    filtered_entities = filtered_entities_df['Entity'].tolist()

    input_df['all_ner'] = input_df['all_ner'].map(convert_and_preprocess)

    entity_counts_matrix = np.array([[entities.count(entity) for entity in filtered_entities] for entities in input_df['all_ner']])
    
    pca_transformed = ipca.transform(entity_counts_matrix)

    n_components = 20
    pca_columns = [f'entity_present_pca_{i}' for i in range(n_components)]
    pca_results = pd.DataFrame(pca_transformed, columns=pca_columns)

    original_columns = input_df.columns.tolist()
    original_columns.remove('all_ner')
    original_data = input_df[original_columns].reset_index(drop=True)

    final_df = pd.concat([original_data, pca_results], axis=1)




    
    for col in final_df.columns:
        if final_df[col].dtype == bool:
            final_df[col] = final_df[col].astype(int)


    with open('feature_columns.json', 'r') as file:
        feature_columns = json.load(file)
    final_df = final_df[feature_columns]

    return final_df

# Example usage:
# df_result = preprocess_data(input_df)