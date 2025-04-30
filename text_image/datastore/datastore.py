import json
from pathlib import Path


class Datastore:
    def __init__(self):
        self._datastore = dict()
        self._reverse_datastore = dict()
        self._query_datastore = dict()

    def insert_or_update_data_dict(self, image_content):
        print(f"[datastore][Start] insert_or_update_data_dict ....")
        file_name = image_content.get_image_file()
        ios = image_content.get_image_objects()
        print(f"[datastore] Filename={file_name} and ios={type(ios)}")
        io_labels = image_content.get_image_object_labels()
        print(f"io_labels = {io_labels}")
        if self._datastore.get(file_name) == None:
            if type(io_labels) == str:
                self._datastore[file_name] = io_labels # we will add a str.
            else:
                raise TypeError("tags should be str type")
        else:
            exist_tags = self._datastore.get(file_name)
            if type(io_labels) == str:
                exist_tags.append(ios)
            self._datastore[file_name] = exist_tags
        print(f"[end] insert_or_update_data_dict.")    


    def show_datastore(self):
        print("\n[datastore] ...")
        for key, value in self._datastore.items():
            print(f"{key} => {value}")
        print("---\n")

    def show_reverse_datastore(self):
        print("[Reverse datastore] ...")
        for key, value in self._reverse_datastore.items():
            print(f"{key} => {value}")
        print("---\n")       

  
    def find_images(self, query_text):
        # We search through the whole dictionary. Linear search -- O(N)
        found_images = []
        for key_text, value_images in self._reverse_datastore.items():
            print(f"key_text={key_text}")
            if query_text in key_text:
                found_images.append(value_images)
        return found_images

    def create_reverse_store(self):
        for key, value in self._datastore.items():
            key = str(key)
            if type(value) == str:
                rev_store_key = self._reverse_datastore.get(value)
                if rev_store_key == None:
                    self._reverse_datastore[value] = [key]
                else:
                    rev_store_key.append(key)
                    self._reverse_datastore[value] = rev_store_key
            else:
                raise TypeError("value should be a string.")


    def save_reverse_datastore(self, file_path):
        # Serializing json
        json_object = json.dumps(self._reverse_datastore, indent=4)
        with open(str(file_path), "w") as outfile:
            outfile.write(json_object)


    def load_json_data(self, file_path):
        # Opening JSON file
        print(f"file_path={file_path}")
        with open(file_path, 'r') as open_file:
            # Reading from json file
            self._query_datastore = json.load(open_file)
        return self._query_datastore
