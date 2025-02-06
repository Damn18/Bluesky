import time
import pandas as pd
import igraph as ig
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import powerlaw
import numpy as np
import seaborn as sns

# Start timer
start_time = time.time()

# ---------------------------
# 1. Load dataset
# ---------------------------
try:
    print("Loading dataset...")
    file_path = "bluesky_processed.snappy.parquet"
    # For testing, we load 10,000 rows; remove slicing when running on the full dataset.
    df = pq.read_table(file_path).to_pandas()[:100]
    print(f"Dataset loaded successfully. Number of rows: {len(df)}")
except Exception as e:
    print("Error loading dataset:", e)
    exit(1)

# ---------------------------
# 2. Build the directed graph 
# ---------------------------
try:
    print("Creating directed graph...")

    # Create a mapping from post URI to author (vectorized)
    uri_to_author = df.set_index("uri")["author"].to_dict()

    # Filter only posts that are replies
    df_filtered = df.dropna(subset=["reply_to"]).copy()

    # Map reply_to to the corresponding author quickly
    df_filtered["target"] = df_filtered["reply_to"].map(uri_to_author)
    df_filtered = df_filtered[df_filtered["target"].notnull()]

    # Group by (author, target) to count interactions (compute edge weights)
    edge_weights_df = (
        df_filtered.groupby(["author", "target"])
                   .size()
                   .reset_index(name="weight")
    )

    # Build the list of all vertices: union of all authors and targets
    authors_from_df = set(df["author"].unique())
    targets = set(df_filtered["target"].unique())
    all_nodes = list(authors_from_df.union(targets))

    # Create mapping from node name to index
    node_to_index = {name: idx for idx, name in enumerate(all_nodes)}

    # Create list of edges (as pairs of vertex indices) and a list of corresponding weights
    edges = []
    weights = []
    for _, row in edge_weights_df.iterrows():
        src = node_to_index.get(row["author"])
        tgt = node_to_index.get(row["target"])
        if src is not None and tgt is not None:
            edges.append((src, tgt))
            weights.append(row["weight"])

    # Create a directed igraph and add vertices and weighted edges
    G = ig.Graph(directed=True)
    G.add_vertices(all_nodes)  # vertices will have attribute "name" equal to each node in all_nodes
    G.add_edges(edges)
    G.es["weight"] = weights

    print(f"Graph created with {G.vcount()} nodes and {G.ecount()} edges.")
except Exception as e:
    print("Error creating graph with igraph:", e)
    exit(1)

# ---------------------------
# 3. Network statistics
# ---------------------------
try:
    # Compute in-degrees and out-degrees (binary counts)
    in_degrees = G.degree(mode="in")
    out_degrees = G.degree(mode="out")
    in_deg_dict = dict(zip(G.vs["name"], in_degrees))
    out_deg_dict = dict(zip(G.vs["name"], out_degrees))

    # Compute reply counts
    authors_without_replies = sum(1 for d in in_degrees if d == 0)
    authors_with_1reply   = sum(1 for d in in_degrees if d == 1)
    authors_with_2replies = sum(1 for d in in_degrees if d == 2)
    authors_with_more     = sum(1 for d in in_degrees if d > 2)

    print(f"Authors with 0 replies: {authors_without_replies}")
    print(f"Authors with 1 reply: {authors_with_1reply}")
    print(f"Authors with 2 replies: {authors_with_2replies}")
    print(f"Authors with >2 replies: {authors_with_more}")

    percentage_without_replies = (authors_without_replies / G.vcount()) * 100
    print(f"Percentage of authors without any reply: {percentage_without_replies:.2f}%")

    # Compute reciprocity and density
    reciprocity = G.reciprocity()
    density = G.density()
    print(f"Reciprocity rate: {reciprocity:.2%}")
    print(f"Network density: {density:.6f}")
except Exception as e:
    print("Error computing network statistics:", e)
    exit(1)



# ---------------------------
# 4. Degree Distribution Analysis 
# ---------------------------
try:
    print("Fitting power-law distribution...")
    in_degree_values = np.array(in_degrees)
    out_degree_values = np.array(out_degrees)

    if len(set(in_degree_values)) > 1 and len(in_degree_values) > 10:
        fit_in = powerlaw.Fit(in_degree_values)
        gamma_in = fit_in.alpha
        ks_in = fit_in.D  # KS distance via the D attribute
        print(f"Power-law exponent (in-degree): {gamma_in:.2f} (KS: {ks_in:.4f})")
    else:
        print("Insufficient unique in-degree values for power-law fitting.")

    if len(set(out_degree_values)) > 1 and len(out_degree_values) > 10:
        fit_out = powerlaw.Fit(out_degree_values)
        gamma_out = fit_out.alpha
        ks_out = fit_out.D
        print(f"Power-law exponent (out-degree): {gamma_out:.2f} (KS: {ks_out:.4f})")
    else:
        print("Insufficient unique out-degree values for power-law fitting.")
except Exception as e:
    print("Error in power-law fitting:", e)
    exit(1)

# ---------------------------
# 5. Compute Centrality Measures 
# ---------------------------
try:
    print("Computing centrality measures using igraph...")

    # Betweenness centrality (using weights, directed)
    betweenness = G.betweenness(directed=True, weights="weight")
    # Closeness centrality; here using mode "IN" (can be adjusted as needed)
    closeness = G.closeness(mode="IN", weights="weight")

    # Map centrality values to dictionaries by vertex name
    betw_dict = dict(zip(G.vs["name"], betweenness))
    close_dict = dict(zip(G.vs["name"], closeness))

    max_in_degree_node = max(in_deg_dict, key=in_deg_dict.get)
    max_out_degree_node = max(out_deg_dict, key=out_deg_dict.get)
    max_betw_node = max(betw_dict, key=betw_dict.get)
    max_close_node = max(close_dict, key=close_dict.get)
  

    print(f"Highest in-degree node: {max_in_degree_node} ({in_deg_dict[max_in_degree_node]} replies)")
    print(f"Highest out-degree node: {max_out_degree_node} ({out_deg_dict[max_out_degree_node]} replies)")
    print(f"Highest betweenness node: {max_betw_node} ({betw_dict[max_betw_node]:.2f})")
    print(f"Highest closeness node: {max_close_node} ({close_dict[max_close_node]:.2f})")

except Exception as e:
    print("Error computing centrality measures:", e)
    exit(1)


# ---------------------------
# 6. Top Authors by Replies 
# ---------------------------
try:
    top_replied_authors = sorted(in_deg_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    top_replied_df = pd.DataFrame(top_replied_authors, columns=["author", "in_degree"])
    top_replied_df["out_degree"] = top_replied_df["author"].map(out_deg_dict)
    top_replied_df["betweenness"] = top_replied_df["author"].map(betw_dict)
    top_replied_df["closeness"] = top_replied_df["author"].map(close_dict)
    top_replied_df.to_csv("top_10_influent_authors.csv", index=False)
    print("Top 10 replied authors saved to 'top_10_influent_authors.csv'.")
except Exception as e:
    print("Error processing top authors:", e)
    exit(1)


# End timer and report total elapsed time
end_time = time.time()
elapsed = end_time - start_time
print(f"SNA analysis complete in {elapsed:.2f} seconds.")
