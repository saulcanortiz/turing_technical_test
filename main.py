from email.mime import image
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
from PIL import Image
import nltk, re, json
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

class Kosmos2:
    def __init__(self, model_name):
        self.model = Kosmos2ForConditionalGeneration.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
    def generate(self, inputs):
        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True, max_new_tokens=64,)
        return generated_ids
    def short_answer(self, generated_text):
        answer_match = re.search(r"Yes|No", generated_text)
        if answer_match:
            answer = answer_match.group()
        else:
            answer = "unknown"
        return answer
    def is_person(self, image):
        prompt = "Are there people?"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        generated_text = self.processor.batch_decode(self.generate(inputs), skip_special_tokens=True)[0]
        caption, _ = self.processor.post_process_generation(generated_text)
        return self.short_answer(caption)
    def is_car(self, image):
        prompt = "Are there cars?"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        generated_text = self.processor.batch_decode(self.generate(inputs), skip_special_tokens=True)[0]
        caption, _ = self.processor.post_process_generation(generated_text)
        return self.short_answer(caption)
    def description(self, image):
        prompt = "<grounding> An image of"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        generated_text = self.processor.batch_decode(self.generate(inputs), skip_special_tokens=True)[0]
        caption, _ = self.processor.post_process_generation(generated_text)
        return caption
    def get_number(self, caption):
        tokens = nltk.word_tokenize(caption)
        tagged_tokens = nltk.pos_tag(tokens)
        for token, tag in tagged_tokens:
            if tag == 'CD':  # Check for cardinal number tag (e.g., 'two')
                return token
        return None
    def number_person(self, image):
        prompt = "How many people in the image?"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        generated_text = self.processor.batch_decode(self.generate(inputs), skip_special_tokens=True)[0]
        caption, _ = self.processor.post_process_generation(generated_text)
        return self.get_number(caption)
    def number_car(self, image):
        prompt = "How many cars in the image?"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        generated_text = self.processor.batch_decode(self.generate(inputs), skip_special_tokens=True)[0]
        caption, _ = self.processor.post_process_generation(generated_text)
        return self.get_number(caption)
    def json_output(self, image_name):
        image = Image.open(image_name)
        data = {"is_person": self.is_person(image), "is_car": self.is_car(image),"description": self.description(image),
        "number_people": self.number_person(image), "number_car":self.number_car(image)}
        return data
