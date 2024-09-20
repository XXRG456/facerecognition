import random
import itertools
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader():
    
    def __init__(self, source_dir: str):
        
        self.__source_dir = source_dir
        self.__classes = os.listdir(self.__source_dir)
        self.__classes_to_dir = self.__find_classes_dir()
        
        
        
    def __find_classes_dir(self):
        
        paths = {}
        for root, dirs, files in os.walk(self.__source_dir, topdown=True):
            for name in dirs:
                paths[name] = os.path.join(root, name)
                
            
        return paths
                              
    def __repr__(self):
        
        return f"DataLoader(source_dir = {self.__source_dir})"
    
    def run(self):
           
        self.IMAGES = {}
        for clss, directory in self.__classes_to_dir.items():
            images = os.listdir(directory)
            print(f"{clss} has {len(images)} images")
            random.shuffle(images)
            self.CLASS_IMAGES = {}
            
            for image in images:
                path = directory + '/' + image
                if os.path.getsize(path):
                    img = cv2.imread(path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.CLASS_IMAGES[image] = img
                    
            self.IMAGES[clss] = self.CLASS_IMAGES
            
        return self.IMAGES
    
    


class PairsDataLoader(DataLoader):
    
    def __init__(self, source_dir: str, resize: tuple, batch_size_target = {'Train': 500, 'Val': 100}, split_size = 0.8):
        super().__init__(source_dir)
        
        self.resize = resize
        
        self.batch_size_train = batch_size_target['Train'] // 10
        self.batch_size_val = batch_size_target['Val'] // 10
        
        self.batch_size_target = {'Train': self.batch_size_train, 'Val': self.batch_size_val}
        
        self.__calc_ratio()
        
        
    def run(self):
        super().run()
        
        self.__preprocess()
        self.__create_class_combinations()
        self.__train_val_split()
        
        self.X_train, self.y_train = self.__create_pairs('Train')
        self.X_val, self.y_val = self.__create_pairs('Val')
        
        self.__split_pairs()
        
        return (self.X_train_1, self.X_train_2, self.y_train), (self.X_val_1, self.X_val_2, self.y_val)
    
        
    def __preprocess(self):
        
        for clss, dictionary in self.IMAGES.items():
            images = []
            for array in dictionary.values():
                array = cv2.resize(array, self.resize).astype('float32') / 255.0
                images.append(array)
            images = np.stack(images, axis = 0)
            self.IMAGES[clss] = images
            
        return self.IMAGES
    

    def __create_class_combinations(self):
        
        self.class_combinations = [[i, j] for i, j in itertools.combinations_with_replacement(list(self.IMAGES.keys()), 2)]
        random.shuffle(self.class_combinations)
        
        return self.class_combinations

    
    def __train_val_split(self):
        
        for clas in self.IMAGES.keys():
            X_train, X_val = train_test_split(self.IMAGES[clas], train_size = 0.7, random_state = 42)
            self.IMAGES[clas] = {'Train':X_train, 'Val':X_val}
            
        return self.IMAGES
    
    
        
    def __create_pairs(self, typ: str):
        
        pairs = []
        pairs_labels = []

        for class_pairs in self.class_combinations:
            if class_pairs[0] == class_pairs[1]: 
                clas = class_pairs[0] 
                batch = self.IMAGES[clas][typ].shape[0]
                combinations = [[i, j] for i,j in itertools.combinations(range(batch),2)]
                random.shuffle(combinations)
                for i in combinations[:self.zeros_ratio[typ]]:
                    
                    pairs.append(self.IMAGES[clas][typ][i])
                    pairs_labels.append(0)
            else:
                clas_1, clas_2 = class_pairs[0], class_pairs[1]
                batch = min(self.IMAGES[clas_1][typ].shape[0], self.IMAGES[clas_2][typ].shape[0])
                combinations = [[i, j] for i,j in itertools.combinations(range(batch), 2)]
                random.shuffle(combinations)
                for i in combinations[:self.ones_ratio[typ]]:
                    
                    pair = np.array([self.IMAGES[clas_1][typ][i[0]], self.IMAGES[clas_2][typ][i[1]]])
                    pairs.append(pair)
                    pairs_labels.append(1)
        
                    
        pairs = np.array(pairs)
        pairs_labels = np.array(pairs_labels).astype("float32")
        
        indicies = np.arange(pairs.shape[0])
        np.random.shuffle(indicies)
        
        pairs = pairs[indicies]
        pairs_labels = pairs_labels[indicies]

        return pairs, pairs_labels
    
    def __calc_ratio(self):
        
        self.ones_ratio = {'Train': int((2 / 5) * 2 * self.batch_size_target['Train']), 
                           'Val': int((2 / 5) * 2 * self.batch_size_target['Val'])}
        
        self.zeros_ratio = {'Train': int((3 / 5) * 2 * self.batch_size_target['Train']), 
                            'Val': int((3 / 5) * 2 * self.batch_size_target['Val'])} 
        
        
    def __split_pairs(self):


        self.X_train_1 = self.X_train[:,0]
        self.X_train_2 = self.X_train[:,1]

        self.X_val_1 = self.X_val[:,0]
        self.X_val_2 = self.X_val[:,1]
        
        return (self.X_train_1, self.X_train_2), (self.X_val_1, self.X_val_2)