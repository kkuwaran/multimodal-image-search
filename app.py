import gradio as gr
import lancedb

from tiny_imagenet_db import search_images
from clip_embedder import CLIPEmbedder



def generate_app(pipeline: CLIPEmbedder, table: lancedb.table.LanceTable) -> gr.Blocks:
    """Generates a Gradio web application for searching and displaying images based on a text query."""

    # Wrap the search_images function to include pipeline and table
    def search(query: str):
        return search_images(pipeline, table, query)


    # Create the Gradio app interface
    with gr.Blocks() as app:
        # Input section: Textbox for query input and submit button
        with gr.Row():
            text_query = gr.Textbox(value="fish", placeholder="Enter a search term", show_label=False, label="Search Query")
            submit_button = gr.Button("Search")

        # Output section: Gallery for displaying the search results
        with gr.Row():
            image_gallery = gr.Gallery(label="Found Images", show_label=False, elem_id="gallery"
                                       ).style(columns=[3], rows=[3], object_fit="contain", height="auto")
        
        # Connect the button click to the search_images function
        submit_button.click(fn=search, inputs=text_query, outputs=image_gallery)

    # Launch the Gradio app with queuing enabled and sharing the app publicly
    app.queue(max_size=1).launch(share=True, debug=True)
    return app