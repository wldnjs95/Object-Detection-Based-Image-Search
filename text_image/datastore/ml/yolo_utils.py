# import cv2
from PIL import Image
from ultralytics import YOLO
from datastore.ml.vit_utils import ViT
from image_data.image_content import ImageContent

LOWEST_SCORE_CONFIDENCE = 0.4

class Yolo:
    def __init__(self, config):
        self.object_detection_model = YOLO("yolo11x.pt")  # initialize model
        self.classification_model = YOLO("yolo11x-cls.pt")  # initialize model
        self.vit=None
        self.config = config
        if self.config.use_vit_in_yolo():
            self.vit = ViT()
    
    def detect_image_content(self, source_image):
        results = self.object_detection_model.predict(source_image, show=False, conf=LOWEST_SCORE_CONFIDENCE)
        # detected_classes = []
        # for result in results:
        #     boxes = result.boxes  # Boxes object for bounding box outputs
        #     print(f"class boxes = {boxes.cls}")
        #     masks = result.masks  # Masks object for segmentation masks outputs
        #     print(f"masks={masks}")
        #     key_points = result.keypoints  # Keypoints object for pose outputs
        #     print(f"key_points={key_points}")
        #     probs = result.probs  # Probs object for classification outputs
        #     print(f"probs={probs}")
        #     obb = result.obb  # Oriented boxes object for OBB outputs
        #     print(f"obb={obb}")
        #     #result.show()  # display to screen
        #     #result.save(filename="result.jpg")  # save to disk
        # 
        # x=ultralytics.engine.results.Boxes object with attributes:
        #     cls: tensor([51.])
        #     conf: tensor([0.3169])
        #     data: tensor([[ 57.1286,  24.6837, 110.6469,  75.1848,   0.3169,  51.0000]])
        #     id: None
        #     is_track: False
        #     orig_shape: (319, 272)
        #     shape: torch.Size([1, 6])
        #     xywh: tensor([[83.8877, 49.9342, 53.5183, 50.5011]])
        #     xywhn: tensor([[0.3084, 0.1565, 0.1968, 0.1583]])
        #     xyxy: tensor([[ 57.1286,  24.6837, 110.6469,  75.1848]])
        #     xyxyn: tensor([[0.2100, 0.0774, 0.4068, 0.2357]])
        
        # Object detection
        image_array = Image.open(source_image)
        image_content = ImageContent(source_image, "Yolo")
        names = self.object_detection_model.names
        #print(type(names))
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            for box in boxes:
                # print(f"box={box}")
                class_index = int(box.cls.cpu().item())
                # print(f"class_index={class_index}")
                class_label = names[class_index]
                # print(f"class_label={class_label}")
                confidence_index = box.conf.cpu().item()
                # print(f"class_label={class_label}, confidence_index={confidence_index}")

                object_box = box.xyxy.cpu().tolist()[0]
                x1 = int(object_box[0])
                y1 = int(object_box[1])
                x2 = int(object_box[2])
                y2 = int(object_box[3])
                crop_area = (x1,y1, x2, y2)
                cutout_image = image_array.crop(crop_area)
                print(f"id(cutout_image) = {id(cutout_image)}")
                # show cutout image only when you debug
                # cutout_image.show()
                if self.vit:
                    self.get_more_image_information(cutout_image, image_content)
                print(f"[Yolo-OD] {class_label}, {confidence_index:.2f}, object_box={object_box}")
                image_content.add_object(class_label, confidence_index, object_box)

        # From classification
        c_results = self.classification_model.predict(source_image)
        # print(f"classification_results={c_results}, len(c_results) = {len(c_results)}")
        for c_result in c_results:
            top5_indexes = c_result.probs.top5
            #print(f"c_result.prob.top5={top5_indexes}")

            cls_probs = c_result.probs.top5conf.cpu().numpy().tolist()
            #print(f"cls_probs = {cls_probs}")

            for index in range(len(top5_indexes)):
                class_label = c_result.names[top5_indexes[index]]
                confidence_index = cls_probs[index]
                # Remove the labels that have very less probabilities.
                if confidence_index >= LOWEST_SCORE_CONFIDENCE:   # 0.4
                    image_content.add_object(class_label, confidence_index)
                
        return image_content


    def get_more_image_information(self, cropped_image, image_content):
        self.vit.explore_image_content(cropped_image, image_content)


"""
    def test_one_image_content(self, source_image):
        results = self.object_detection_model.predict(source_image, show=False, conf=0.3)
        # detected_classes = []
        # for result in results:
        #     boxes = result.boxes  # Boxes object for bounding box outputs
        #     print(f"class boxes = {boxes.cls}")
        #     masks = result.masks  # Masks object for segmentation masks outputs
        #     print(f"masks={masks}")
        #     key_points = result.keypoints  # Keypoints object for pose outputs
        #     print(f"key_points={key_points}")
        #     probs = result.probs  # Probs object for classification outputs
        #     print(f"probs={probs}")
        #     obb = result.obb  # Oriented boxes object for OBB outputs
        #     print(f"obb={obb}")
        #     #result.show()  # display to screen
        #     #result.save(filename="result.jpg")  # save to disk
        # 
        # x=ultralytics.engine.results.Boxes object with attributes:
        #     cls: tensor([51.])
        #     conf: tensor([0.3169])
        #     data: tensor([[ 57.1286,  24.6837, 110.6469,  75.1848,   0.3169,  51.0000]])
        #     id: None
        #     is_track: False
        #     orig_shape: (319, 272)
        #     shape: torch.Size([1, 6])
        #     xywh: tensor([[83.8877, 49.9342, 53.5183, 50.5011]])
        #     xywhn: tensor([[0.3084, 0.1565, 0.1968, 0.1583]])
        #     xyxy: tensor([[ 57.1286,  24.6837, 110.6469,  75.1848]])
        #     xyxyn: tensor([[0.2100, 0.0774, 0.4068, 0.2357]])
        
        # Object detection

        image_content = ImageContent(source_image)
        names = self.object_detection_model.names
        #print(type(names))
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            for box in boxes:
                # print(f"box={box}")
                class_index = int(box.cls.cpu().item())
                # print(f"class_index={class_index}")
                class_label = names[class_index]
                # print(f"class_label={class_label}")
                confidence_index = box.conf.cpu().item()
                # print(f"class_label={class_label}, confidence_index={confidence_index}")
                image_content.add_object(class_label, confidence_index)
                coords = box.xyxy.cpu().tolist()[0]
                #cutout = image[y1:y2, x1:x2]
                # print(f"coords={coords}")

        # From classification
        c_results = self.classification_model.predict(source_image)
        # print(f"classification_results={c_results}, len(c_results) = {len(c_results)}")
        for c_result in c_results:
            top5_indexes = c_result.probs.top5
            # print(f"c_result.prob.top5={top5_indexes}")

            cls_probs = c_result.probs.top5conf.cpu().numpy().tolist()
            # print(f"cls_probs = {cls_probs}")

            for index in range(len(top5_indexes)):
                class_label = c_result.names[top5_indexes[index]]
                confidence_index = cls_probs[index]
                image_content.add_object(class_label, confidence_index)
                
        return image_content
""" 