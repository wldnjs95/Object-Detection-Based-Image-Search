import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from PIL import Image
from datastore.ml.unique_tags import HUMAN_INPUT_CLASSES
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from image_data.image_content import ImageContent

class Grounding_Dino:
    def __init__(self):
        self.model_id = "IDEA-Research/grounding-dino-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Grounding Dino is running on", self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
        
    def detect_image_content(self, source_image):
        image = Image.open(source_image).convert("RGB")
        
        image_content = ImageContent(source_image, "Grounding_Dino")
        
        for key,value in HUMAN_INPUT_CLASSES.items():
            text = f"{value}."
            #print("[Grounding_Dino]","key=", key, "value=",value)
            inputs = self.processor(images=image, text=text, return_tensors="pt")
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=0.6,
                    target_sizes=[image.size[::-1]]
                )
            except Exception as e:
                #if 'threshold' in str(e):
                #    print("Threshold is not supported in this model, using box_threshold instead")
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.5,
                    text_threshold=0.5,
                    target_sizes=[image.size[::-1]]
                )

            try:
                for idx, (label, score, box) in enumerate(zip(
                    results[0]['text_labels'],
                    results[0]['scores'],
                    results[0]['boxes']
                )):
                    image_content.add_object(label, score, box)
            except Exception as e:
                #print("[NOTI] ",e, "was not working using 'labels' instead")
                for idx, (label, score, box) in enumerate(zip(
                    results[0]['labels'],
                    results[0]['scores'],
                    results[0]['boxes']
                )):
                    image_content.add_object(label,score,box)
            
        return image_content


def main():
    grounding_dino = Grounding_Dino()
    test_image = "/Users/jiwonpark/Desktop/2025/Capstone/code/0324_prof/text_image/images/jiwon_image/camera_1.JPG"
    grounding_dino.detect_image_content(test_image)

if __name__ == "__main__":
    main()
