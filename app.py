import gradio as gr
import pandas as pd
import os
from PIL import Image

# Load the CSV file
data_path = "/content/drive/MyDrive/001_projects/FSL/FSL_2025/Pseudo_Label/UB/sprint_4/final_fsl_19_march_dataframe_2_PayToProvOrgName.parquet"
df = pd.read_parquet(data_path)

# Image folder path
img_path = '/content/drive/MyDrive/001_projects/FSL/FSL_2025/Processing_stage_2(Img_Json)/UB/04_Mar/IMG'
KEY_NAME = "2_PayToProvOrgName"

global_index = 0  # Current index
crop_dimension = (500, 0, 1200, 300)


def load_image_and_data(index):
    global global_index
    global_index = index

    row = df.iloc[index]
    image_name = row['Image_Name']
    annotation = row[KEY_NAME]
    image_path = os.path.join(img_path, image_name)

    if os.path.exists(image_path):
        img = Image.open(image_path).crop(crop_dimension)
    else:
        img = None

    progress_text = f"{index + 1} / {len(df)}"
    return img, annotation, progress_text, image_name

def update_annotation(new_annotation):
    df.at[global_index, KEY_NAME] = new_annotation
    df.to_parquet(data_path, index=False)
    return "Annotation saved!"

def next_image():
    new_index = min(global_index + 1, len(df) - 1)
    return load_image_and_data(new_index)

def prev_image():
    new_index = max(global_index - 1, 0)
    return load_image_and_data(new_index)

def go_to_index(index):
    index = max(0, min(index, len(df) - 1))
    return load_image_and_data(index)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_display = gr.Image()
        with gr.Column():
            annotation_label = gr.Label(value=f"Key: {df.columns[0]}")
            image_name_display = gr.Textbox(label="Image Name", interactive=False)
            annotation_text = gr.Textbox(label="Annotation")
            progress_display = gr.Textbox(label="Progress", interactive=False)
            index_input = gr.Number(label="Go to Index", value=0)
            save_button = gr.Button("Save Annotation")
            save_status = gr.Textbox(label="Status", interactive=False)
            next_button = gr.Button("Next Image")
            prev_button = gr.Button("Previous Image")
            go_button = gr.Button("Go")

    save_button.click(update_annotation, inputs=[annotation_text], outputs=[save_status])
    next_button.click(next_image, outputs=[image_display, annotation_text, progress_display, image_name_display])
    prev_button.click(prev_image, outputs=[image_display, annotation_text, progress_display, image_name_display])
    go_button.click(go_to_index, inputs=[index_input], outputs=[image_display, annotation_text, progress_display, image_name_display])

    # Load the first image on start
    image, annotation, progress, image_name = load_image_and_data(global_index)
    image_display.value = image
    annotation_text.value = annotation
    progress_display.value = progress
    image_name_display.value = image_name

if __name__ == '__main__':
    demo.launch(share = True)