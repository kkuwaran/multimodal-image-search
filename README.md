# Multimodal Image Search (multimodal-image-search)

**Multimodal Image Search** is a Python project designed to enable efficient image search using natural language queries. 
It combines the power of [CLIP](https://huggingface.co/docs/transformers/en/model_doc/clip) for text and image embeddings, 
[Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) as a dataset, and 
[LanceDB](https://lancedb.com/) for fast vector search. 
A user-friendly interface is built with [Gradio](https://gradio.app/).


## Features
- **Multimodal Embeddings:** Extract text and image embeddings using CLIP.
- **Search with Text Queries:** Retrieve similar images using natural language input.
- **Efficient Storage:** Store and search embeddings with LanceDB.
- **Interactive Interface:** Visualize search results through a Gradio-powered web app.
- **Batch Processing:** Efficiently process and embed datasets for large-scale applications.


## Prerequisites
- Python 3.8 or later
- CUDA-enabled GPU (optional for faster processing)


## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/kkuwaran/multimodal-image-search.git
   cd multimodal-image-search
   ```
2. **Download the Tiny ImageNet dataset:**
   The dataset will be automatically downloaded and loaded using the `datasets` library during execution.


## Usage
1. **Setting Up the Environment** <br>
   Edit the `main.ipynb` file to configure the model name, dataset split, and table name:
   ```python
   model_name = "openai/clip-vit-base-patch32"  # Choose the desired model
   data_split = "valid"  # Choose either 'train' or 'valid' dataset split
   table_name = "image_search"  # Specify the table name for storing embeddings
   ```
2. **Processing the Dataset and Running the Application** <br>
   Run the `main.ipynb` notebook to:
   * Load the Tiny ImageNet dataset.
   * Generate image embeddings using the CLIP model.
   * Store the processed embeddings in a LanceDB table for efficient retrieval.
   * Launch a Gradio app for a simple and interactive user interface.


## Project Structure

```bash
📦 multimodal-image-search
 ├── main.ipynb            # Notebook for dataset processing and visualization
 ├── app.py                # User interface of the Gradio app
 ├── tiny_imagenet_db.py   # Dataset loading and interaction with LanceDB
 └── clip_embedder.py      # Utilities for generating CLIP-based embeddings
```
