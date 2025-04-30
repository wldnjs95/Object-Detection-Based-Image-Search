from pathlib import Path
from collections import OrderedDict

class Config:
    def __init__(self):
        self.config_map = OrderedDict()

    def check_starts_width_dot(self, any_path):
        if any_path.startswith("."):
            current_path = str(Path.cwd())
            config_file = Path(current_path + any_path[1:])
            print(f"config_file={config_file}")
        else:
            config_file = Path(current_path + any_path[0:])
        return config_file

    def set_config_map(self, config_data_file):
        print(f"config_data_file = {config_data_file}")
        self.config_map["BASE_DIR"] = str(Path.cwd())
        config_file = self.check_starts_width_dot(config_data_file)
        self.config_map["CONFIG_FILE"] = config_file
        print(config_file.exists())    
        with open(str(config_file), "r") as cf:
            lines = cf.readlines()
            #print(lines)
            for line in lines:
                line = line.strip()
                if len(line) == 0 or line.startswith("#"):
                    continue
                words = line.split("=")
                print(f"words = {words}, {len(words)}")
                if len(words) == 2:
                    tag = words[0].strip()
                    value = words[1].strip()
                    # print(f"tag={tag},value={value}")
                    if value.startswith("./"):
                        self.config_map[tag.upper()] = self.check_starts_width_dot(value)
                    else:
                        print(f"else value= {value}")
                        self.config_map[tag] = value
                else:
                    raise ValueError("Words can not be more than 2.")

    def show_configuration(self):
        print("="*10)
        print("   Config map")
        print("="*10)
        for key, value in self.config_map.items():
            print(f" {key}=>{value}")
        print("="*10)

    def get_store_option(self):
        return self.config_map["STORE_MODE"]
    
    def get_query_option(self):
        return self.config_map["QUERY_MODE"]
    
    def get_images_storage_dir(self):
        return self.config_map["IMAGE_STORAGE_DIR"]

    def get_persist_storage_dir(self):
        return self.config_map["PERSIST_STORAGE_DIR"]

    def use_yolo(self):
        return self.config_map["USE_YOLO"]=="True"

    def use_detr(self):
        return self.config_map["USE_DETR"]=="True"
    
    def use_vit_in_yolo(self):
        return self.config_map["USE_VIT_IN_YOLO"]=="True"

    def use_vit(self):
        return self.config_map["USE_VIT"]=="True"

    def use_detr_panoptic(self):
        return self.config_map["USE_DETR_PANOPTIC"]=="True"

    def use_grounding_dino(self):
        return self.config_map["USE_GROUNDING_DINO"] =="True"

