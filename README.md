# **The Dawn of a New Blue(sky) Era: Structural and Content Analysis of the Growing Social Media**
This repository contains the code and methodology for analyzing the **interaction structure** and **content** of the Bluesky social media platform. The study is part of the **Computational Social Science (CSS)** course and applies **Social Network Analysis (SNA)** and **Topic Modeling (BERTopic)** to understand user interactions and discussion patterns.

The dataset consists of **1 million public posts** collected from the Hugging Face repository:  
📌 [alpindale/two-million-bluesky-posts](https://huggingface.co/datasets/alpindale/two-million-bluesky-posts)

---

## **⚠ Important Note**  
Due to **space constraints**, the following files are **not included in this repository**:
- **The trained BERTopic model**
- **The preprocessed dataset (`bluesky_processed.snappy.parquet`)**

🚨 **If you are not processing the dataset locally, you must store it manually.**  
📩 If needed, **contact me** to obtain the zipped dataset and model, otherwise, the **SNA script will not work**.

---

## **Installation Instructions**
To run this project, you need to install all dependencies listed in `requirements.txt`.

### **1️⃣ Install Required Libraries**

pip install -r requirements.txt

### **2️⃣ Run functions.py ** 

This script defines all necessary functions used across the pipeline:

    Text preprocessing (lemmatization, regex cleaning)
    Language detection
    Parallelized processing for efficiency

### **3️⃣ Run preprocessing_berTopic.py ** 

This script performs:

    Dataset loading
    Filtering for English posts
    Text preprocessing and lemmatization
    Running BERTopic for topic modeling
    Saving the processed dataset (bluesky_processed.snappy.parquet)

This step will output:
    Processed dataset
    BERTopic model

(If you don't process this locally, request the zipped dataset as mentioned earlier.)

### **4️⃣ Run python sna.py ** 


This script performs Social Network Analysis, including:

    Graph construction
    Centrality metrics 
    Community detection using the Louvain algorithm
    Power-law fitting for degree distribution



