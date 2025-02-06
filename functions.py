
import re
import contractions
import spacy
import langid
from langid import classify
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Precompiled regex patterns for efficiency
URL_REGEX = re.compile(r"http[s]?://\S+|www\.\S+")
NON_ALPHA_REGEX = re.compile(r"[^a-zA-Z\s]")
WHITESPACE_REGEX = re.compile(r"\s+")

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------------

def preprocess_text(text):
    """(Legacy function kept for reference – not used in the new pipeline.)"""
    try:
        text = contractions.fix(text)  # Expand contractions
        text = URL_REGEX.sub("", text)  # Remove URLs
        text = NON_ALPHA_REGEX.sub("", text)  # Remove punctuation and digits
        text = WHITESPACE_REGEX.sub(" ", text).strip()  # Collapse multiple spaces
        doc = nlp(text.lower())  # Process text with SpaCy
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_digit]
        return " ".join(tokens)
    except Exception as e:
        return f"ERROR: {e}"

def is_english(text):
    """Check if a text is in English using langid."""
    try:
        lang, _ = classify(text)
        return lang == "en"
    except Exception:
        return False
    
# ---------------------------------------------------------------------

def parallel_language_detection(texts, num_workers=None, chunksize=1000):
    """
    Use multiprocessing Pool to perform language detection.
    A chunksize is provided to reduce inter-process communication overhead.
    """
    if num_workers is None:
        num_workers = cpu_count()  # Use all available CPUs
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(is_english, texts, chunksize=chunksize), total=len(texts)))
    return results

# ---------------------------------------------------------------------

def parallel_preprocessing(texts, n_process=None, batch_size=1000):
    """
    Efficient text preprocessing using spaCy's nlp.pipe.
    A generator cleans each text (expanding contractions and applying precompiled regex),
    and then nlp.pipe processes the texts in batches and in parallel.
    """
    if n_process is None:
        n_process = cpu_count()  # Use all available CPUs

    def clean_text_generator(texts):
        """Generator that yields cleaned text for spaCy processing."""
        for text in texts:
            try:
                cleaned = contractions.fix(text)
                cleaned = URL_REGEX.sub("", cleaned)
                cleaned = NON_ALPHA_REGEX.sub("", cleaned)
                cleaned = WHITESPACE_REGEX.sub(" ", cleaned).strip()
                yield cleaned.lower()
            except Exception as e:
                yield f"ERROR: {e}"

    processed_texts = []
    # Use spaCy's built-in multiprocessing (n_process > 1) and batching
    for doc in tqdm(nlp.pipe(clean_text_generator(texts), batch_size=batch_size, n_process=n_process),
                    total=len(texts)):
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_digit]
        processed_texts.append(" ".join(tokens))
    return processed_texts

