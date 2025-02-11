import pandas as pd
import re
import time
import spacy
from datasets import load_dataset
from langid import classify  
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from bertopic import BERTopic
import pyarrow as pa
import pyarrow.parquet as pq
import os 
import contractions
import matplotlib.pyplot as plt
from functions import preprocess_text, is_english, parallel_language_detection, parallel_preprocessing

start_time = time.time()

print("Loading dataset...")
try:
    ds = load_dataset("alpindale/two-million-bluesky-posts")
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Pandas DataFrame
print("Converting dataset to DataFrame...")
df = pd.DataFrame(ds['train'][:1000000])
print("Conversion completed.")

# Dataset size
initial_count = len(df)

# Dataset filtering
print("Selecting required columns...")
df = df[['text', 'author', 'reply_to', 'uri']]
print("Columns selected.")

df = df[df["text"].str.len() > 10]

# ---------------------------------------------------------------------
# Language filtering
print("Filtering posts in English...")
try:
    df['is_english'] = parallel_language_detection(df['text'].tolist())
    df = df[df['is_english']].drop(columns=['is_english'])
    print("Filtering completed.")
except Exception as e:
    print(f"Error filtering language: {e}")
    exit()

# ---------------------------------------------------------------------
# Text preprocessing using the efficient spaCy pipeline
print("Executing text preprocessing...")
try:
    df['processed_text'] = parallel_preprocessing(df['text'].tolist())
    df["text"] = df["processed_text"]
    print("Preprocessing completed.")
except Exception as e:
    print(f"Error in preprocessing")
    exit()

final_count = len(df)
preprocessing_reduction = ((initial_count - final_count) / initial_count) * 100
print(f"Number of posts before preprocessing: {initial_count}")
print(f"Number of posts after preprocessing: {final_count}")
print(f"Reduction percentage: {preprocessing_reduction:.2f}%")

# ---------------------------------------------------------------------
# Topic Modeling with BERTopic
print("Starting Topic Modeling with BERTopic...")
try:
    model_path = os.path.join(os.getcwd(), "bertopic_model")
    topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2")
    
    # Fit-transform on the preprocessed text
    topics, _ = topic_model.fit_transform(df["processed_text"].dropna().tolist())
    # Directly assign topic numbers to the DataFrame
    df["label"] = topics
    
    topic_model.save(model_path)
    print(f"BERTopic model saved at: {model_path}")
    print("BERTopic modeling completed.")
except Exception as e:
    print(f"Error in BERTopic execution: {e}")

print(f"Total number of topics: {len(topic_model.get_topics())}")

# ---------------------------------------------------------------------
# Outlier count
outlier_count = df[df["label"] == -1].shape[0]
print(f"Outlier count is: {outlier_count}")

# 10 topics
filtered_df = df[df["label"] != -1]  # (Remove outliers)
topic_counts = filtered_df["label"].value_counts().head(10)
top_10_topics = topic_counts.index.tolist()
print(f"Top 10 topics are: {top_10_topics}")

topic_descriptions = {}
for topic in top_10_topics:
    topic_words = topic_model.get_topic(topic)
    if isinstance(topic_words, list):
        topic_descriptions[topic] = ", ".join([word[0] for word in topic_words[:4]])
    else:
        topic_descriptions[topic] = "Unknown"

# Create DataFrame with topic frequencies
output_df = pd.DataFrame({
    "Topic": [f"{k}_{topic_descriptions[k]}" if k in topic_descriptions else str(k) for k in topic_counts.index],
    "Count": topic_counts.values
})

# ---------------------------------------------------------------------
# Dataset in Parquet format
print("Saving dataset to working directory...")
try:
    output_path = os.path.join(os.getcwd(), "bluesky_processed.snappy.parquet")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression='snappy')
    print(f"Saving completed. File saved at: {output_path}")
except Exception as e:
    print(f"Error saving dataset: {e}")

# ---------------------------------------------------------------------
# Preview of pre-processed data
print("Preview of pre-processed data:")
print(df.head(15))

# Distribution
plt.figure(figsize=(10, 5))
plt.barh(output_df["Topic"].head(10), output_df["Count"].head(10), color="steelblue")
plt.xlabel("Frequency")
plt.ylabel("Topics")
plt.title("Top 10 Topics Distribution")
plt.gca().invert_yaxis()  # Invert y-axis for better readability

plot_path = os.path.join(os.getcwd(), "top_topics_distribution.png")
plt.savefig(plot_path, bbox_inches='tight')
print(f"Plot saved at: {plot_path}")

# ---------------------------------------------------------------------
print("Generating BERTopic heatmap visualization...")
fig_2 = topic_model.visualize_heatmap(top_n_topics=10)

# BERTopic visualization
heatmap_path = os.path.join(os.getcwd(), "bertopic_heatmap.png")
fig_2.write_image(heatmap_path)
print(f"BERTopic bar chart saved at: {heatmap_path}")

# Preview of topic counts
print("Preview of topic counts:")
print(output_df.head(15))

# ---------------------------------------------------------------------
total_time = time.time() - start_time
print(f"Total execution time: {total_time:.2f} seconds")

