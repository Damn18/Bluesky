# **Bluesky: Structural and Content Analysis**
This repository contains the code and methodology for analyzing the **interaction structure** and **content** of the Bluesky social media platform. The study is part of the **Computational Social Science (CSS)** course and applies **Social Network Analysis (SNA)** and **Topic Modeling (BERTopic)** to understand user interactions and discussion patterns.

## **Overview**
This research aims to answer two primary questions:
1. **What is the structure of interaction networks on Bluesky?**  
   - Using **SNA**, we analyze centrality measures, degree distributions, and community detection.
2. **What are the main topics discussed on Bluesky?**  
   - Using **BERTopic**, we extract and cluster key topics discussed on the platform.

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
```bash
pip install -r requirements.txt

This installs essential libraries such as:

    pandas
    spacy
    datasets
    langid
    bertopic
    tqdm
    matplotlib
    powerlaw
    pyarrow
    networkx
    igraph

Additionally, ensure that the required SpaCy language model is installed:

python -m spacy download en_core_web_sm

Execution Steps

Follow these steps to process the dataset and perform SNA and topic modeling:
2️⃣ Run functions.py

This script defines all necessary functions used across the pipeline:

    Text preprocessing (lemmatization, regex cleaning)
    Language detection
    Parallelized processing for efficiency

python functions.py

(This step ensures that helper functions are available before running other scripts.)
3️⃣ Run preprocessing_berTopic.py

This script performs:

    Dataset loading
    Filtering for English posts
    Text preprocessing and lemmatization
    Running BERTopic for topic modeling
    Saving the processed dataset (bluesky_processed.snappy.parquet)

To execute:

python preprocessing_berTopic.py

This step will output:

    Processed dataset
    BERTopic model

(If you don't process this locally, request the zipped dataset as mentioned earlier.)
4️⃣ Run sna.py

This script performs Social Network Analysis, including:

    Graph construction (nodes = users, edges = replies)
    Centrality metrics (in-degree, out-degree, betweenness)
    Community detection using the Louvain algorithm
    Power-law fitting for degree distribution

To execute:

python sna.py

📌 Note: If the processed dataset is missing, this step will fail.
Outputs and Results

    The processed dataset will be saved in Parquet format (bluesky_processed.snappy.parquet).
    SNA analysis outputs include:
        Network metrics (degree distribution, centrality measures)
        Community detection results
        Power-law fitting statistics
    BERTopic results include:
        Topic clusters
        Heatmap visualization
        Top 10 most discussed topics

References

For more details, see the full paper:
📄 The Dawn of a New Blue(sky) Era: Structural and Content Analysis
Contact

If you need the preprocessed dataset or the trained BERTopic model, please contact me.
Otherwise, the SNA analysis will not work without the required dataset.

📩 Author: Damiano Orlandi


### **🔹 Why This README is Useful?**
✅ **Clear step-by-step execution guide**  
✅ **Installation instructions for required dependencies**  
✅ **Warnings about missing large files and how to obtain them**  
✅ **Brief explanation of the research questions and methodologies used**  

This README ensures that **anyone** can replicate the res
