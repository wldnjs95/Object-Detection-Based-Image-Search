from image_data.image_object import ImageObject
import re
#from image_object import ImageObject

DELIMITER = "+"

class ImageContent:
    def __init__(self, image_file, model_name = None):
        self.pretrained_model_name = model_name
        self.objects = []
        self.image_file = image_file

    def normalize_class_label(self, label):
        label = re.sub(r'[^a-z0-9]', '_', label.lower())
        label = re.sub(r'_+', '_', label)
        return label

    def add_object(self, object_label, object_probability=-1, object_box = None):
        image_object = ImageObject(object_label, object_probability, object_box)
        self.objects.append(image_object)

    def get_image_file(self):
        return self.image_file

    def get_image_objects(self):
        return self.objects

    def get_image_object_labels(self):
        object_labels = ""
        print(self.objects)
        for obj in self.objects:
            cleaned_label = self.normalize_class_label(obj.get_label())
            object_labels += cleaned_label + DELIMITER
        object_labels = object_labels[:(len(object_labels)-len(DELIMITER))]
        return object_labels
            

    def show(self):
        print(f"[Image Content] {'='*30} ")
        print(f"image file = {self.image_file}")
        print(self.objects)
        for obj in self.objects:
            obj.show()
        print("="*50)

    def __str__(self):
        objs = ""
        for obj in self.objects:
            objs = objs + str(obj) + ", "
        return objs[:len(objs)-2]

# test = ImageContent("A")
# test.add_object("a",0.3)
# test.add_object("b",0.8)
# test.show()
# print(test)