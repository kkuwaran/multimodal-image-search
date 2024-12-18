import io
from typing import List

import pandas as pd
import PIL.Image
from datasets import DatasetDict, Dataset
from datasets import load_dataset
import lancedb
from lancedb.pydantic import LanceModel, vector

from clip_embedder import CLIPEmbedder



def load_tiny_imagenet(data_split: str = 'train', verbose: bool = False) -> Dataset:
    """Load the Tiny ImageNet dataset for the specified split."""

    assert data_split in ['train', 'valid'], "Invalid data_split. Must be 'train' or 'valid'."

    datasets: DatasetDict = load_dataset("zh-plus/tiny-imagenet")
    dataset = datasets[data_split]
    if verbose:
        print(f"Dataset info:\n{dataset}\n")
    return dataset



class Image(LanceModel):
    """A model for storing and processing image data in LanceDB."""

    image: bytes
    label: int
    vector: vector(512)


    def to_pil(self) -> PIL.Image.Image:
        """Convert the image bytes into a PIL Image."""

        return PIL.Image.open(io.BytesIO(self.image))


    @staticmethod
    def pil_to_bytes(img: PIL.Image.Image) -> bytes:
        """Convert a PIL Image to bytes."""

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()  # image data in bytes format


    @classmethod
    def create_table(cls, table_name: str, dataset: Dataset) -> lancedb.table.LanceTable:
        """Create a LanceDB table from a given dataset."""

        # Connect to the LanceDB database
        db = lancedb.connect("~/.lancedb")
        # Drop existing table with the same name (if any)
        db.drop_table(table_name, ignore_missing=True)
        # Create a new table with the schema derived from the class
        table = db.create_table(table_name, schema=cls.to_arrow_schema())

        # Validate the dataset keys
        required_keys = {"image", "label", "vector"}
        assert set(dataset.features.keys()) == required_keys, f"Dataset must contain keys: {required_keys}"

        # Create Image instances and prepare data for the table
        image_instances = []
        for data in dataset:
            data["image"] = Image.pil_to_bytes(data["image"])
            image_instances.append(cls(**data))

        # Add data to the table
        table.add(image_instances)
        return table



def search_images(clip_embedder: CLIPEmbedder, table: lancedb.table.LanceTable, 
                  query: str, verbose: bool = False) -> List[PIL.Image.Image]:
    """Search for the closest images in the table based on a text query."""

    # Generate the embedding for the text query
    query_embedding = clip_embedder.embed_text([query]).numpy()

    # Search the table for the top 9 closest matches
    search_results = table.search(query_embedding).limit(9)
    if verbose:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 500)
        print(search_results.to_pandas())
    
    # Convert search results to pydantic Image objects
    image_results = search_results.to_pydantic(Image)

    # Convert image bytes to PIL Image objects for visualization
    images_pil = [image.to_pil() for image in image_results]
    return images_pil