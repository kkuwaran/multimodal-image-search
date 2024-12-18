import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast



class CLIPEmbedder:
    """A class to handle text and image embedding using a pre-trained CLIP model."""
    
    def __init__(self, model_name: str):
        """Initializes the CLIPEmbedder with the specified pre-trained model."""

        self.model_name = model_name
        
        # Select device (CUDA if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        # Load the model, processor, and tokenizer
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)


    def embed_text(self, text_batch: list) -> torch.Tensor:
        """
        Generate text embeddings for a batch of texts using the CLIP model.
        - text_batch: A list of text strings to embed.
        """

        # Tokenize the input text batch
        inputs = self.tokenizer(text_batch, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)

        # Generate text embeddings without calculating gradients
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs)  # shape: (batch_size, 512)
        
        # Move embeddings to CPU and remove singleton dimensions
        text_embeddings = text_embeddings.squeeze().cpu()
        return text_embeddings


    def embed_image(self, image_batch: list) -> torch.Tensor:
        """
        Generate image embeddings for a batch of images using the CLIP model.
        - image_batch: A list of images to embed (PIL images or tensors).
        """

        # Preprocess the image batch and convert it to the required format
        inputs = self.processor(text=None, images=image_batch, return_tensors="pt")
        inputs = inputs.to(self.device)  # type: dict

        # Generate image embeddings without calculating gradients
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)  # shape: (batch_size, 512)
        
        # Move embeddings to CPU and remove singleton dimensions
        image_embeddings = image_embeddings.squeeze().cpu()
        return image_embeddings