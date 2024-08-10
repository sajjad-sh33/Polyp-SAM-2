from torch.utils.data import Dataset
import json
import numpy as np
from numpy import random
from PIL import Image
import cv2
import glob



#######################################################################################################################
class kvasir(Dataset):

    def __init__(self, n_points_add, n_points_rmv, transform=None ):
        
        with open('./Kvasir for segmentation/kvasir-seg/kavsir_bboxes.json', 'r') as f:
            self.file = json.load(f)
        self.root_dir = list(self.file.keys())
        self.n_points_add = n_points_add
        self.n_points_rmv = n_points_rmv
        self.transform = transform

    def __len__(self):
        return len(self.root_dir)



    def __getitem__(self, idx):

        img_pth = "./Kvasir for segmentation/kvasir-seg/images/" + self.root_dir[idx] + ".jpg"
        msk_pth = "./Kvasir for segmentation/kvasir-seg/masks/" + self.root_dir[idx] + ".jpg"
        
        image = Image.open(img_pth)
        image = np.array(image.convert("RGB"))
        
        mask_np = cv2.imread(msk_pth, cv2.IMREAD_GRAYSCALE)/255
        mask_np = (mask_np>0.5)*1
        
        if self.transform:
            image = self.transform(image)
         


        bboxes = self.file[self.root_dir[idx]]['bbox']
        
        num_masks = len(bboxes)
                        
        point_coords = np.zeros((num_masks, self.n_points_add + self.n_points_rmv, 2))
        point_labels = np.zeros((num_masks, self.n_points_add + self.n_points_rmv))
        bounding_box = np.zeros((num_masks, 4))
        

        for x in range(num_masks):
            mask = np.zeros((mask_np.shape), dtype=np.uint8) # initialize mask
            mask[bboxes[x]['ymin']:bboxes[x]['ymax'],bboxes[x]['xmin']:bboxes[x]['xmax']] = 1 
            mask = mask * mask_np

            ori_add = np.where( mask == 1)
            ori_rmv = np.where( mask == 0)

            
            rand_add = random.randint(ori_add[0].shape[0], size = self.n_points_add)
            rand_rmv = random.randint(ori_rmv[0].shape[0], size = self.n_points_rmv)
            
            
            for i in range(self.n_points_add):
                point_coords[x, i] = (ori_add[1][rand_add[i]], ori_add[0][rand_add[i]])
                point_labels[x, i] = 1
                
            for i in range(self.n_points_rmv):
                point_coords[x, i + self.n_points_add] = (ori_rmv[1][rand_rmv[i]], ori_rmv[0][rand_rmv[i]])
                point_labels[x, i + self.n_points_add] = 0

            bounding_box[x] = (bboxes[x]['xmin'], bboxes[x]['ymin'], bboxes[x]['xmax'], bboxes[x]['ymax'])
            
    

        return {"image" : image,
                "mask_np" : mask_np,
                "point_coords" : point_coords,
               "point_labels" : point_labels,
                "bounding_box" : bounding_box,
                }









#######################################################################################################################
class CVC_300(Dataset):

    def __init__(self, n_points_add, n_points_rmv, transform=None ):
        
        self.root_dir = glob.glob('./CVC-300/images/*.png')
        self.n_points_add = n_points_add
        self.n_points_rmv = n_points_rmv
        self.transform = transform

    def __len__(self):
        return len(self.root_dir)



    def __getitem__(self, idx):

        img_pth = self.root_dir[idx]
        msk_pth = './CVC-300/masks/' + img_pth.split('/')[-1]

        image = Image.open(img_pth)
        image = np.array(image.convert("RGB"))
        
        mask_np = cv2.imread(msk_pth, cv2.IMREAD_GRAYSCALE)/255
        
        if self.transform:
            image = self.transform(image)
         


        seg_value = 1.
        segmentation = np.where(mask_np == seg_value)
        
        # Bounding Box
        bboxes = 0, 0, 0, 0
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))
        
            bboxes = x_min, y_min, x_max, y_max
                
            num_masks = 1
                        
        point_coords = np.zeros((num_masks, self.n_points_add + self.n_points_rmv, 2))
        point_labels = np.zeros((num_masks, self.n_points_add + self.n_points_rmv))
        bounding_box = np.zeros((num_masks, 4))
        


        ori_add = np.where( mask_np == 1)
        ori_rmv = np.where( mask_np == 0)

        
        rand_add = random.randint(ori_add[0].shape[0], size = self.n_points_add)
        rand_rmv = random.randint(ori_rmv[0].shape[0], size = self.n_points_rmv)
        
        
        for i in range(self.n_points_add):
            point_coords[0, i] = (ori_add[1][rand_add[i]], ori_add[0][rand_add[i]])
            point_labels[0, i] = 1
            
        for i in range(self.n_points_rmv):
            point_coords[0, i + self.n_points_add] = (ori_rmv[1][rand_rmv[i]], ori_rmv[0][rand_rmv[i]])
            point_labels[0, i + self.n_points_add] = 0

        bounding_box[0] = (bboxes[0], bboxes[1], bboxes[2], bboxes[3])
    

        return {"image" : image,
                "mask_np" : mask_np,
                "point_coords" : point_coords,
               "point_labels" : point_labels,
                "bounding_box" : bounding_box,
                }




#######################################################################################################################
class CVC_ClinicDB(Dataset):

    def __init__(self, n_points_add, n_points_rmv, transform=None ):
        
        self.root_dir = glob.glob('./CVC-ClinicDB/images/*.png')
        self.n_points_add = n_points_add
        self.n_points_rmv = n_points_rmv
        self.transform = transform

    def __len__(self):
        return len(self.root_dir)



    def __getitem__(self, idx):

        img_pth = self.root_dir[idx]
        msk_pth = './CVC-ClinicDB/masks/' + img_pth.split('/')[-1]

        image = Image.open(img_pth)
        image = np.array(image.convert("RGB"))
        
        mask_np = cv2.imread(msk_pth, cv2.IMREAD_GRAYSCALE)/255
        
        if self.transform:
            image = self.transform(image)
         


        seg_value = 1.
        segmentation = np.where(mask_np == seg_value)
        
        # Bounding Box
        bboxes = 0, 0, 0, 0
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))
        
            bboxes = x_min, y_min, x_max, y_max
                
            num_masks = 1
                        
        point_coords = np.zeros((num_masks, self.n_points_add + self.n_points_rmv, 2))
        point_labels = np.zeros((num_masks, self.n_points_add + self.n_points_rmv))
        bounding_box = np.zeros((num_masks, 4))
        


        ori_add = np.where( mask_np == 1)
        ori_rmv = np.where( mask_np == 0)

        
        rand_add = random.randint(ori_add[0].shape[0], size = self.n_points_add)
        rand_rmv = random.randint(ori_rmv[0].shape[0], size = self.n_points_rmv)
        
        
        for i in range(self.n_points_add):
            point_coords[0, i] = (ori_add[1][rand_add[i]], ori_add[0][rand_add[i]])
            point_labels[0, i] = 1
            
        for i in range(self.n_points_rmv):
            point_coords[0, i + self.n_points_add] = (ori_rmv[1][rand_rmv[i]], ori_rmv[0][rand_rmv[i]])
            point_labels[0, i + self.n_points_add] = 0

        bounding_box[0] = (bboxes[0], bboxes[1], bboxes[2], bboxes[3])
    

        return {"image" : image,
                "mask_np" : mask_np,
                "point_coords" : point_coords,
               "point_labels" : point_labels,
                "bounding_box" : bounding_box,
                }

#######################################################################################################################
class CVC_ColonDB(Dataset):

    def __init__(self, n_points_add, n_points_rmv, transform=None ):
        
        self.root_dir = glob.glob('./CVC-ColonDB/images/*.png')
        self.n_points_add = n_points_add
        self.n_points_rmv = n_points_rmv
        self.transform = transform

    def __len__(self):
        return len(self.root_dir)



    def __getitem__(self, idx):

        img_pth = self.root_dir[idx]
        msk_pth = './CVC-ColonDB/masks/' + img_pth.split('/')[-1]

        image = Image.open(img_pth)
        image = np.array(image.convert("RGB"))
        
        mask_np = cv2.imread(msk_pth, cv2.IMREAD_GRAYSCALE)/255
        
        if self.transform:
            image = self.transform(image)
         


        seg_value = 1.
        segmentation = np.where(mask_np == seg_value)
        
        # Bounding Box
        bboxes = 0, 0, 0, 0
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))
        
            bboxes = x_min, y_min, x_max, y_max
                
            num_masks = 1
                        
        point_coords = np.zeros((num_masks, self.n_points_add + self.n_points_rmv, 2))
        point_labels = np.zeros((num_masks, self.n_points_add + self.n_points_rmv))
        bounding_box = np.zeros((num_masks, 4))
        


        ori_add = np.where( mask_np == 1)
        ori_rmv = np.where( mask_np == 0)

        
        rand_add = random.randint(ori_add[0].shape[0], size = self.n_points_add)
        rand_rmv = random.randint(ori_rmv[0].shape[0], size = self.n_points_rmv)
        
        
        for i in range(self.n_points_add):
            point_coords[0, i] = (ori_add[1][rand_add[i]], ori_add[0][rand_add[i]])
            point_labels[0, i] = 1
            
        for i in range(self.n_points_rmv):
            point_coords[0, i + self.n_points_add] = (ori_rmv[1][rand_rmv[i]], ori_rmv[0][rand_rmv[i]])
            point_labels[0, i + self.n_points_add] = 0

        bounding_box[0] = (bboxes[0], bboxes[1], bboxes[2], bboxes[3])
    

        return {"image" : image,
                "mask_np" : mask_np,
                "point_coords" : point_coords,
               "point_labels" : point_labels,
                "bounding_box" : bounding_box,
                }


#######################################################################################################################
class etis(Dataset):

    def __init__(self, n_points_add, n_points_rmv, transform=None ):
        
        self.root_dir = glob.glob('./etis/images/*.png')
        self.n_points_add = n_points_add
        self.n_points_rmv = n_points_rmv
        self.transform = transform

    def __len__(self):
        return len(self.root_dir)



    def __getitem__(self, idx):

        img_pth = self.root_dir[idx]
        msk_pth = './etis/masks/' + img_pth.split('/')[-1]

        image = Image.open(img_pth)
        image = np.array(image.convert("RGB"))
        
        mask_np = cv2.imread(msk_pth, cv2.IMREAD_GRAYSCALE)/255
        
        if self.transform:
            image = self.transform(image)
         


        seg_value = 1.
        segmentation = np.where(mask_np == seg_value)
        
        # Bounding Box
        bboxes = 0, 0, 0, 0
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))
        
            bboxes = x_min, y_min, x_max, y_max
                
            num_masks = 1
                        
        point_coords = np.zeros((num_masks, self.n_points_add + self.n_points_rmv, 2))
        point_labels = np.zeros((num_masks, self.n_points_add + self.n_points_rmv))
        bounding_box = np.zeros((num_masks, 4))
        


        ori_add = np.where( mask_np == 1)
        ori_rmv = np.where( mask_np == 0)

        
        rand_add = random.randint(ori_add[0].shape[0], size = self.n_points_add)
        rand_rmv = random.randint(ori_rmv[0].shape[0], size = self.n_points_rmv)
        
        
        for i in range(self.n_points_add):
            point_coords[0, i] = (ori_add[1][rand_add[i]], ori_add[0][rand_add[i]])
            point_labels[0, i] = 1
            
        for i in range(self.n_points_rmv):
            point_coords[0, i + self.n_points_add] = (ori_rmv[1][rand_rmv[i]], ori_rmv[0][rand_rmv[i]])
            point_labels[0, i + self.n_points_add] = 0

        bounding_box[0] = (bboxes[0], bboxes[1], bboxes[2], bboxes[3])
    

        return {"image" : image,
                "mask_np" : mask_np,
                "point_coords" : point_coords,
               "point_labels" : point_labels,
                "bounding_box" : bounding_box,
                }


