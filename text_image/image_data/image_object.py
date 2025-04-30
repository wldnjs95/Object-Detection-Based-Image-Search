
class ImageObject:
    def __init__(self, label, prob, box = None): 
        self.prob = prob
        self.label = label
        self.box = box

    def show(self):
        print(f' {"-"*30}')
        print(f"    label = {self.label}, prob = {self.prob}, box = {self.box}")
        print(f' {"-"*30}')

    def __str__(self):
        return f" {self.label}: {self.prob}, {self.box}"

    def get_label(self):
        return self.label
    
    def get_conf(self):
        return self.prob