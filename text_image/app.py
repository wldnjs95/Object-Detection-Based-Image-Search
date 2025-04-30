import gradio as gr
from main import Driver

driver = Driver()

def run_store():
    driver.store()
    return "Image Data Processed and Stored"

def run_search(query_text):
    image_paths = driver.retrieve(query_text)
    if not image_paths:
        return gr.update(value=f"{query_text} images doesn't exist"), []
    image_paths = [str(path) for path in image_paths]
    return gr.update(value="Search complete."), image_paths

with gr.Blocks() as demo:
    gr.Markdown("## Smart Image Store & Search")

    gr.HTML("""
    <style>
    .gallery-item img {
        max-height: 150px !important;
        object-fit: contain !important;
    }
    </style>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            save_btn = gr.Button("Store Image Data")
            query_textbox = gr.Textbox(label="Query Text")
            search_btn = gr.Button("Search")
            result_box = gr.Textbox(label="Status", lines=2)

        with gr.Column(scale=2):
            image_gallery = gr.Gallery(label="Search Results", columns=4, height="auto", object_fit="contain")

    save_btn.click(fn=run_store, outputs=result_box)
    search_btn.click(fn=run_search, inputs=query_textbox, outputs=[result_box, image_gallery])

demo.launch()