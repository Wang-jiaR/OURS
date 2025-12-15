import os
import pickle
import json
from collections import OrderedDict
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class COCO2017(DatasetBase):

    dataset_dir = "coco2017"

    def __init__(self, cfg):
        print(cfg.DATASET.ROOT)
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        
        # 检查是否有按类别组织的目录结构
        self.organized_dir = os.path.join(self.dataset_dir, "images_organized")
        if os.path.exists(self.organized_dir):
            # 使用按类别组织的目录结构
            self.image_dir = self.organized_dir
            self.use_organized_structure = True
        else:
            # 使用原始目录结构
            self.image_dir = os.path.join(self.dataset_dir, "images")
            self.use_organized_structure = False
            
        self.annotations_dir = os.path.join(self.dataset_dir, "annotations")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            
            if self.use_organized_structure:
                # 使用按类别组织的目录结构
                train = self.read_data_organized(classnames, "train")
                test = self.read_data_organized(classnames, "test")
            else:
                # 使用原始目录结构
                train = self.read_data(classnames, "train2017", "instances_train2017.json")
                test = self.read_data(classnames, "val2017", "instances_val2017.json")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <class_id>: <class_name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                class_id = line[0]
                classname = " ".join(line[1:])
                classnames[class_id] = classname
        return classnames

    def read_data_organized(self, classnames, split_dir):
        """Read data from organized directory structure (by class)."""
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for folder in folders:
            # folder name is the class ID (0-79)
            label = int(folder)
            if str(label) in classnames:
                classname = classnames[str(label)]
                folder_path = os.path.join(split_dir, folder)
                imnames = listdir_nohidden(folder_path)
                for imname in imnames:
                    impath = os.path.join(folder_path, imname)
                    item = Datum(impath=impath, label=label, classname=classname)
                    items.append(item)

        return items

    def read_data(self, classnames, split_dir, annotation_file):
        """Read data from COCO format annotations."""
        split_dir = os.path.join(self.image_dir, split_dir)
        annotation_path = os.path.join(self.annotations_dir, annotation_file)
        
        # 读取COCO标注文件
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)
        
        # 创建image_id到file_name的映射
        image_id_to_file = {}
        for image_info in coco_data['images']:
            image_id_to_file[image_info['id']] = image_info['file_name']
        
        # 创建image_id到labels的映射
        image_id_to_labels = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            category_id = annotation['category_id']
            if image_id not in image_id_to_labels:
                image_id_to_labels[image_id] = []
            image_id_to_labels[image_id].append(category_id)
        
        items = []
        for image_id, file_name in image_id_to_file.items():
            if image_id in image_id_to_labels:
                # 只处理有标注的图片
                labels = image_id_to_labels[image_id]
                # 对于多标签情况，取第一个标签（或者你可以根据需要修改这个逻辑）
                label = labels[0] - 1  # COCO类别ID从1开始，转换为从0开始
                
                if str(label) in classnames:
                    impath = os.path.join(split_dir, file_name)
                    classname = classnames[str(label)]
                    item = Datum(impath=impath, label=label, classname=classname)
                    items.append(item)
        
        return items 