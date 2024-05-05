from main import Kosmos2
import gradio as gr
import json


 
def image_processor(image_filename):
    service = Kosmos2(model_name="microsoft/kosmos-2-patch14-224")
    result  = service.json_output(image_filename)
    return json.dumps(result)
 
iface = gr.Interface(fn=image_processor, inputs="text", outputs="text", title="Image Processor")
iface.launch()