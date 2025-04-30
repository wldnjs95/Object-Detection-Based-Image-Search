from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
from image_data.image_content import ImageContent

LOWEST_SCORE_CONFIDENCE = 0.4

class ViT:
    def __init__(self):
        model_name = "google/vit-base-patch16-224"
        self.classification_model = ViTForImageClassification.from_pretrained(model_name)  # initialize model
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.image_file_name = None
 
    def detect_image_content(self, source_image):
        img_1 = Image.open(source_image).convert("RGB")
        # print(f"shape of image = {img_1.size}")
        inputs = self.processor(images=img_1, return_tensors="pt")
        outputs = self.classification_model(**inputs)

        logits = outputs.logits
        # print(f"logits={logits}")
        # print(f"logits.argmax(-1)={logits.argmax(-1)}")
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class_name = self.classification_model.config.id2label[predicted_class_idx]
        # print(f"[ViT] Predicted class: {predicted_class_name}")

        conf = torch.max(torch.nn.functional.softmax(logits))
        # print(f"[ViT] conf = {conf}")
        image_content = ImageContent(source_image, "Vit")
    
        if type(conf) == torch.Tensor:
            conf = conf.cpu().item()

        image_content.add_object(predicted_class_name, conf)
        # print(f"[ViT][explore_image_content] class: {predicted_class_name} conf = {conf}")

        return image_content

    def explore_image_content(self, cutout_image, image_content):
        img_1 = cutout_image.convert("RGB")
        # print(f"shape of image = {img_1.size}")
        inputs = self.processor(images=img_1, return_tensors="pt")
        outputs = self.classification_model(**inputs)

        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class_name = self.classification_model.config.id2label[predicted_class_idx]
        conf = torch.max(torch.nn.functional.softmax(logits))
        
        if type(conf) == torch.Tensor:
            conf = conf.cpu().item()

        if conf >= LOWEST_SCORE_CONFIDENCE:
            image_content.add_object(predicted_class_name, conf)
        print(f"[ViT][explore_image_content] class: {predicted_class_name} conf = {conf}")

        # return image_content

def main():
    vit = ViT()
    test_image_1 = "/Users/jiwonpark/Desktop/2025/Capstone/code/0324_prof/text_image/images/jiwon_image/camera_1.JPG"
    # test_image_2 = "C:\\Users\\YSR\\MuniShounak\\jiwon_capstone\\projects\\text_image\\images\\water\\water_2.png"
    vit.detect_image_content(test_image_1)

if __name__ == "__main__":
    main()
