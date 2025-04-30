from config.base_config import Config
from datastore.image_store import ImageStoreSystem
from retrieve.image_retrieval import ImageRetrievalSystem
import time

import os
import torch

torch.cuda.empty_cache()
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:32"

class Driver:
    def __init__(self):
        self.config = Config()
        self.config.set_config_map("./config.cfg")
        self.config.show_configuration()

    def store(self):
        start_time = time.time()
        image_store_sys = ImageStoreSystem(self.config)
        image_store_sys.collect_all_image_paths()
        image_store_sys.process_images()
        image_store_sys.store_images()
        image_store_sys.save_file()
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Time taken to store {len(image_store_sys.all_image_files)} images: {elapsed:.2f} seconds")

    def retrieve(self, query_text = None):
        image_retrieval_sys = ImageRetrievalSystem(self.config)
        image_retrieval_sys.load_query_database()
        image_paths = image_retrieval_sys.get_images_with_text(query_text)
        image_retrieval_sys.show_images(image_paths)
        return image_paths

    def store_or_retrieve(self, action):
        print(f"action = {action}")
        if action == self.config.get_store_option():
            self.store()        
        elif action == self.config.get_query_option():
            self.retrieve()
        else:
            print(f"action not defined = {action}")


    def main(self):
        if True:
            action = self.config.get_store_option()
        else:
            action = self.config.get_query_option()
        self.store_or_retrieve(action)



if __name__ == "__main__":
    driver = Driver()
    driver.main()
