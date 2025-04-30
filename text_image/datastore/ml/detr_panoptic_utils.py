from PIL import Image
import torch
from transformers import DetrFeatureExtractor, DetrForSegmentation
import requests
from image_data.image_content import ImageContent

# COCO GITHUB FILE: https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json

class Detr_Panoptic:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Detr_Panoptic is running on ", self.device)
        self.feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101-panoptic')
        self.model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-101-panoptic').to(self.device)
        
        # COCO Panoptic Categories JSON
        self.url = "https://raw.githubusercontent.com/cocodataset/panopticapi/master/panoptic_coco_categories.json"
        self.coco_panoptic_categories = requests.get(self.url).json()
        self.coco_category_map = {category["id"]: category["name"] for category in self.coco_panoptic_categories}

    
    def detect_image_content(self, source_image):
        image = Image.open(source_image).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        
        processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0).to(self.device)
        result = self.feature_extractor.post_process_panoptic(outputs, processed_sizes, threshold=0.5)[0]
                

        # Note
        # score, box exists only for 'isthing=True'
        image_content = ImageContent(source_image, "Detr_Panoptic")
        
        for seg in result['segments_info']:
            cat_id = seg['category_id']
            label_name = self.coco_category_map[cat_id]
            image_content.add_object(label_name)
        #     is_thing = seg['isthing']
        #     #print(f"Area: {seg['area']}")
        #     #print(f"ID: {seg['id']}")
        return image_content
        
        
    
def main():
    detr_panoptic = Detr_Panoptic()
    test_image = "/Users/jiwonpark/Desktop/2025/Capstone/code/0324_prof/text_image/images/jiwon_image/camera_1.JPG"
    test_image2 = "/Users/jiwonpark/Desktop/2025/Capstone/code/0324_prof/text_image/images/jiwon_image/avo.JPG"
    detr_panoptic.detect_image_content(test_image2)

if __name__ == "__main__":
    main()
