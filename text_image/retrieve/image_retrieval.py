# 1) Store an image
# 2) query with a text

import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
import json
from collections import Counter
from pathlib import Path
from config.base_config import Config
from datastore.datastore import Datastore


class ImageRetrievalSystem:
    def __init__(self, config):
        self.config = config
        self.images_dir_path = self._set_images_dir_path()
        self.store_dir_path = self._set_store_dir_path()
        self._query_database = self.load_query_database()

    def _set_images_dir_path(self):
        images_dir_path = Path(self.config.get_images_storage_dir())
        if not images_dir_path.exists():
            images_dir_path.mkdir()
        print(f"images_dir_path ={images_dir_path}")
        return images_dir_path
    
    def _set_store_dir_path(self):
        persist_dir_path = Path(self.config.get_persist_storage_dir())
        store_file_name = "/store.json"
        store_dir_path = Path(str(persist_dir_path) + store_file_name)
        print(f"store_dir_path ={store_dir_path}")
        return store_dir_path

    def load_query_database(self):
        # Opening JSON file
        print(f"file_path={self.store_dir_path}")
        with open(self.store_dir_path, 'r') as open_file:
            # Reading from json file
            query_database = json.load(open_file)
        return query_database

    def get_images_with_text(self, query_text= None):
        if not query_text:
            query_text = input("Enter a text:").strip()
        print(f"You are looking for: {query_text}")
        image_paths = self.find_image_paths(query_text)
        if image_paths == None:
            print(f"There is no image for the [{query_text}]")
            return []
        else:
            print(f"The images are:")
            count  = 1
            for image_path in image_paths:
                print(f"[{count}]  {image_path}")
                count += 1
            print()
        return image_paths

    def find_image_paths(self, query_text):
        # We search through the whole dictionary. Linear search -- O(N)
        found_images = []
        for key_text, value_images in self._query_database.items():
            print(f"\n query_text = {query_text}, key_text={key_text}, value_images={value_images}")
            # if query_text in key_text:
            #     print("found ....{query_text}")
            #     found_images.extend(value_images)
            if self.check_all_words_in_key_text(query_text, key_text):
                if value_images not in found_images:
                    found_images.extend(value_images)
        return found_images


    def check_all_words_in_key_text(self, query_text, key_text):
        # words = query_text.split("+")
        words = Counter(query_text.split("+"))
        keys_count = Counter(key_text.split("+"))
        for word, count in words.items():
            if keys_count[word] < count:
                return False
        return True
            

    def show_images(self, image_paths):

        if len(image_paths) == 0:
            print("There are no images to show.")
            return
        # Load images using PIL (or scikit-image):
        image_arrays = []
        for image_path in image_paths:
            img = Image.open(image_path)
            image_arrays.append(img)
        
        image_array_len = len(image_arrays)
        if image_array_len == 1:
            fig = plt.figure()
            plt.imshow(image_arrays[0], cmap='gray' if np.array(image_arrays[0]).ndim == 2 else None)
        else:
            fig, axes = plt.subplots(1,image_array_len, figsize=(10, 5)) # Adjust as needed
            for i, image_array in enumerate(image_arrays):
                axes[i].imshow(image_array, cmap='gray' if np.array(image_array).ndim == 2 else None) # Use 'gray' for grayscale
                axes[i].axis('off') # Hide axes
                axes[i].set_title(f"Image {i+1}")
            
        plt.show()
        