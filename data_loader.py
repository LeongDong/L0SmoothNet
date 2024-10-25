from os.path import splitext
from os import listdir
from PIL import Image
from pre_processed import hwc_to_chw, img_Normalize, mask_Normalize
import numpy as np

'''
return file names of all images in the same directory in order
'''
def get_ids(dir):
    filename_sort = listdir(dir)
    filename_sort.sort(key=lambda x:int(x[:-4]))
    return (splitext(file)[0] for file in filename_sort)

'''
return cropped images by pre-set size
'''
def to_get_image(ids,dir,suffix,height=256,width=256):
    for id in ids:
        img = Image.open(dir+id+suffix)
        gray_img = img.convert('L')
        gray_img = gray_img.resize((height,width),Image.ANTIALIAS)

        gray_img = np.array(gray_img,dtype=np.float32)
        image = gray_img
        yield image

def get_images_ids(ids,dir_img):
    imgs = to_get_image(ids,dir_img,'.png')
    img_change = map(hwc_to_chw,imgs)
    img_normalized = map(img_Normalize,img_change)

    return zip(ids,img_normalized)

def to_crop_image_aug(ids,dir_image,dir_mask,suffix):
    for id in ids:
        image = Image.open(dir_image+id+suffix)
        image = np.array((image.convert('L')), dtype=np.int)

        mask = Image.open(dir_mask+id+suffix)
        mask = np.array((mask.convert('L')), dtype=np.int)

        image = hwc_to_chw(image)
        image = img_Normalize(image)
        mask = hwc_to_chw(mask)
        mask = mask_Normalize(mask)
        image_mask = np.concatenate((image,mask),axis=0)
        yield image_mask

def get_images_train(ids,dir_image,dir_mask):
    image_mask = to_crop_image_aug(ids,dir_image,dir_mask,'.png')

    return image_mask

