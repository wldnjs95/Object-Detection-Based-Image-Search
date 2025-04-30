from pathlib import Path

import numpy as np
import torch
import json

from datastore.datastore import Datastore
from datastore.ml.yolo_utils import Yolo
from datastore.ml.detr_utils import Detr
from datastore.ml.vit_utils import ViT
from datastore.ml.detr_panoptic_utils import Detr_Panoptic
from datastore.ml.grounding_dino import Grounding_Dino

from image_data.image_content import ImageContent

class ImageStoreSystem:
    def __init__(self, config):
        self.config = config
        self._datastore = Datastore()
        self.images_dir_path = None
        self.store_dir_path = None
        
        # Model var
        self.yolo = None
        self.detr = None
        self.vit = None
        self.detr_panoptic = None
        self.grounding_dino = None

        # Model init
        if self.config.use_yolo():
            self.yolo = Yolo(self.config)
        if self.config.use_detr():
            self.detr = Detr()
        if self.config.use_vit():
            self.vit = ViT()
        if self.config.use_detr_panoptic():
            self.detr_panoptic = Detr_Panoptic()
        if self.config.use_grounding_dino():
            self.grounding_dino = Grounding_Dino()
        self._set_images_dir_path()
        self._set_store_dir_path()
        self.all_image_files = []

    def _set_images_dir_path(self):
        self.images_dir_path = Path(self.config.get_images_storage_dir())
        if not self.images_dir_path.exists():
            self.images_dir_path.mkdir()
        print(f"self.images_dir_path ={self.images_dir_path}")
    
    def _set_store_dir_path(self):
        persist_dir_path = Path(self.config.get_persist_storage_dir())
        store_file_name = "/store.json"
        self.store_dir_path = Path(str(persist_dir_path) + store_file_name)
        print(f"self.store_dir_path ={self.store_dir_path}")

    # def _add_image_to_database(self, image_name, tags):
    #     print(f"add an image {image_name} and tags={tags}")
    #     self._datastore.insert_or_update_data(image_name, tags)

    def collect_all_image_paths(self):
        # start reading image files
        file_paths = self.images_dir_path.iterdir()
        print("Loading image files ...")
        for file_path in file_paths:
            if str(file_path.name).startswith("."):
                continue
            print(file_path)
            self.all_image_files.append(file_path)
        print(f"Done loading {len(self.all_image_files)}images.")

    def process_images(self):
        count = 1
        for img_file in self.all_image_files:
            print(f"\n{'='*80}\n==>>> [{count}] process {img_file}")
            # object detectors
            image_content_yolo = None
            if self.yolo:
                image_content_yolo = self.yolo.detect_image_content(img_file)
                print(f"[Yolo]: image_content_yolo = {image_content_yolo}")

            image_content_detr = None
            if self.detr:
                image_content_detr = self.detr.detect_image_content(img_file)
                print(f"[Detr]: image_content_detr = {image_content_detr}")

            image_content_vit = None
            if self.vit:
                image_content_vit = self.vit.detect_image_content(img_file)
                print(f"[Vit]: image_content_vit = {image_content_vit}")

            image_content_detr_panoptic = None
            if self.detr_panoptic:
                print("[log-jw] DETR PANOPTIC RUNNING ")
                print("[low-jw] config.use_detr_panoptic() = ", self.config.use_detr_panoptic())
                print("[log-jw] type of config value", type(self.config.use_detr_panoptic()))
                image_content_detr_panoptic = self.detr_panoptic.detect_image_content(img_file)
                print(f"[Detr Panoptic]: image_content_detr_panoptic = {image_content_detr_panoptic}")
            image_content_grounding_dino = None
            if self.grounding_dino:
                image_content_grounding_dino = self.grounding_dino.detect_image_content(img_file)
                print(f"[Grounding Dino]: image_content_grounding_dino = {image_content_grounding_dino}")

            final_image_content = self.merge_model_ensemble(
                img_file,
                image_content_yolo,
                image_content_detr,
                image_content_vit,
                image_content_detr_panoptic,
                image_content_grounding_dino,
            )
            # self._show_image_content(final_image_content)
            self._add_image_content_to_database(final_image_content)
            count += 1

    def _add_image_content_to_database(self, image_content):
        print(f"** add an image_content to datastore.")
        self._datastore.insert_or_update_data_dict(image_content)

    def _show_image_content(self, image_content):
        print(f"image_contents = {image_content}")
        image_content.show()

    # need to complete this part.
    def store_images(self):
        self._datastore.show_datastore()
        self._datastore.create_reverse_store()
        self._datastore.show_reverse_datastore()

    def save_file(self):
        print(f"storing store_dir_path={self.store_dir_path}")
        self._datastore.save_reverse_datastore(self.store_dir_path)


    # def merge_model_ensemble(self, image_content_yolo, image_content_detr, image_content_vit):
    #     print(f"[Merge Models] {'-'*50}")
    #     found_objects = {}
    #     for yolo_image_object in image_content_yolo.get_image_objects():
    #         print(yolo_image_object)
    #         self.update_found_objects_dict(yolo_image_object, found_objects)
    #     for detr_image_object in image_content_detr.get_image_objects():
    #         print(detr_image_object)
    #         self.update_found_objects_dict(detr_image_object, found_objects)
    #     for vit_image_object in image_content_vit.get_image_objects():
    #         print(vit_image_object)
    #         self.update_found_objects_dict(vit_image_object, found_objects)
    #     print("*"*10)
    #     print (f"found_objects={found_objects}")
    #     print("*"*10)
    #     print(f"[Merge Models] {'-'*50}")  


    # def update_found_objects_dict(self, image_object, found_objects):
    #     label = image_object.get_label()
    #     conf = image_object.get_conf()
    #     print(f"type(conf)={type(conf)}")
    #     if type(conf) == torch.Tensor:
    #         conf = conf.cpu().item()

    #     if found_objects.get(label) == None: 
    #         found_objects[label] = conf
    #     else:
    #         found_objects[label] = np.mean(found_objects.get(label) + conf)

    def merge_model_ensemble(self, img_file, image_content_yolo, image_content_detr, image_content_vit, image_content_detr_panoptic, image_content_grounding_dino):
        print(f"[Merge Models] {'-'*50}")

        merged_image_content = ImageContent(img_file) 

        yolo_labels = []
        if image_content_yolo is not None:
            for yolo_image_object in image_content_yolo.get_image_objects():
                merged_image_content.add_object(
                    yolo_image_object.get_label(), yolo_image_object.get_conf()
                )
                yolo_labels.append(yolo_image_object.get_label())

        if image_content_detr is not None:
            for detr_image_object in image_content_detr.get_image_objects():
                if detr_image_object.get_label() not in yolo_labels: 
                    merged_image_content.add_object(
                        detr_image_object.get_label(), detr_image_object.get_conf()
                    )

        if image_content_vit is not None:
            for vit_image_object in image_content_vit.get_image_objects():
                merged_image_content.add_object(
                    vit_image_object.get_label(), vit_image_object.get_conf()
                )
            
        if image_content_detr_panoptic is not None:
            for detr_panoptic_image_object in image_content_detr_panoptic.get_image_objects():
                merged_image_content.add_object(
                    detr_panoptic_image_object.get_label(), detr_panoptic_image_object.get_conf()
                )
        
        if image_content_grounding_dino is not None:
            for grounding_dino_image_object in image_content_grounding_dino.get_image_objects():
                merged_image_content.add_object(
                    grounding_dino_image_object.get_label(), grounding_dino_image_object.get_conf()
                )
            
        print(f"merged_image_content={merged_image_content}")
        print(f"[Merge Models] {'-'*50}")  
        return merged_image_content