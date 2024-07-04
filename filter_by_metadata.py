import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import cosine_similarity
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

def compute_cosine_similarity(text1, text2):
    text1 = str(text1)
    text2 = str(text2)
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)[0][1]
    return cosine_sim

def filter_attributes(metadata_entry, key, value):
    if (key=='title'):
        cos_sim = compute_cosine_similarity(metadata_entry['title'], value)
        return cos_sim*10
    elif (key == 'author'):
        return 1.0 if value in metadata_entry['author'] else 0.0
    elif (key == 'abstract'):
        cos_sim = compute_cosine_similarity(metadata_entry['abstract'], value)
        return cos_sim*10
    elif (key == 'keywords'):
        cos_sim = compute_cosine_similarity(metadata_entry['abstract'], value)
        return cos_sim*10
    elif (key == 'publication_date'):
        op = value[0] if value[1].isdigit() else value[0:2]
        value = value[len(op):]
        filter_date = datetime.strptime(value, "%Y-%m-%d")
        if metadata_entry['publication_date'] == "N/A":
            return 0.0
        entry_date = datetime.strptime(metadata_entry['publication_date'], "%Y-%m-%d")
        if (op == '>'):
            return 2.0 if entry_date > filter_date else -6.0
        elif (op == '>='):
            return 2.0 if entry_date >= filter_date else -6.0
        elif (op == '<'):
            return 2.0 if entry_date < filter_date else -6.0
        elif (op == '<='):
            return 2.0 if entry_date <= filter_date else -6.0
        else:
            return 2.0 if entry_date == filter_date else -6.0
    elif (key == 'results'):
        if (type(metadata_entry['results'])==list):
            metadata_entry['results'] = " ".join(metadata_entry['results'])
        cos_sim = compute_cosine_similarity(metadata_entry['results'], value)
        return cos_sim*10
    else:
        return 0.0

def filter_data(metadata, filter_dict):
    scored_metadata = []
    for entry in metadata:
        total_score = 0.0
        for key, value in filter_dict.items():
            if key in entry:
              total_score += filter_attributes(entry, key, value)
        print(total_score)
        scored_metadata.append((total_score, entry))

    scored_metadata.sort(reverse=True, key=lambda x: x[0])
    top_results = [entry for _, entry in scored_metadata[:3]]
    return top_results


