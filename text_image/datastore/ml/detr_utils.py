from PIL import Image
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from image_data.image_content import ImageContent

LOWEST_SCORE_CONFIDENCE = 0.4

class Detr:
    def __init__(self):
        model_name = "facebook/detr-resnet-101"
        self.object_detection_model = DetrForObjectDetection.from_pretrained(model_name)  # initialize model
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.image_file_name = None
 
    def detect_image_content(self, source_image):
        img_1 = Image.open(source_image).convert("RGB")
        # print(f"shape of image = {img_1.size}")
        inputs = self.processor(images=img_1, return_tensors="pt")
        outputs = self.object_detection_model(**inputs)

        # print(f"[img_1.size[::-1]]={img_1.size}")
        target_sizes = torch.tensor([img_1.size[::-1]])
        results = self.processor.post_process_object_detection(outputs,
                                                               target_sizes=target_sizes,
                                                               threshold=0.9)[0]
        image_content = ImageContent(source_image, "Detr")
        # for score,label, box in zip(results["scores"],
        #                             results["labels"],
        #                             results["boxes"]):
        #     box = [round(i,2) for i in box.tolist()]
        #     detections = []
        #     detection = {
        #         "label" : self.object_detection_model.config.id2label[label.item()],
        #         "conf": round(score.item(),3),
        #         "box" : box
        #     }
        #     detections.append(detection)
        #     image_object
        # print(detections)

        for score,label, box in zip(results["scores"],
                                    results["labels"],
                                    results["boxes"]):
            box = [round(i,2) for i in box.tolist()]
            label = self.object_detection_model.config.id2label[label.item()]
            conf = round(score.item(),3)
            if type(conf) == torch.Tensor:
                conf = conf.cpu().item()

            if conf >= LOWEST_SCORE_CONFIDENCE:
                image_content.add_object(label, conf, box)
        return image_content


def main():
    detr = Detr()
    test_image = "/Users/jiwonpark/Desktop/2025/Capstone/code/0324_prof/text_image/images/jiwon_image/avo.JPG"
    # test_image = "C:\\Users\\YSR\\MuniShounak\\jiwon_capstone\\projects\\text_image\\images\\water\\water_1.png"
    print(detr.detect_image_content(test_image))

if __name__ == "__main__":
    main()
