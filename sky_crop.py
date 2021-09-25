
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.9")   # please manually install torch 1.9 if Colab changes its default version
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
def crop_sky_seg(im):
     #cv2_imshow(im)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))#load panoptic segmentation model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1 # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")#load parameters
    #cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    # get the id of category "sky" from COCO database (for every session's run, this id is different, but the category_id for sky is identical (40))
    info_dict=outputs["panoptic_seg"][1]

    sky_id=0

    for index in range (len(info_dict)):
      if info_dict[index]['category_id']==40: # sky's category_id
        sky_id=info_dict[index]['id']
        break
    print('sky id',sky_id)
    #not waste time to visualize
    '''
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=2) #this "scale" needs to be tuned to fit the photo
    # if the photo is so small, you may want the scale to be bigger and vice versa
    out = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"),outputs["panoptic_seg"][1])
    cv2_imshow(out.get_image()[:, :, ::-1]) # show panoptic segmentation
    '''
    # crop the sky in the original image
    import numpy as np
    id_mat=outputs["panoptic_seg"][0].to("cpu").numpy()
    crop_i=0
    for index in range(id_mat.shape[0]):

     if not (np.any(id_mat[index]==sky_id)):  #exhaustive checking of sky, if any pixel belongs to "sky" in a row
     # that row will be cut
      crop_i=index  #this is last row of sky or the boundary of the crop
      break
    #print('crop i is:',crop_i,id_mat.shape[0])

    im=im[crop_i:,:]
    # Run exactly the same instance segmentation for cropped photo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")) #load the model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1 # set threshold for this model




    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")#load parameters
    #cfg.MODEL.DEVICE = "cpu" # Googlecolab stopped me from using cuda and asked me for using premium version :), 
    #I therefore switched to non cuda (cpu), it is much slower but no option
    # you can (and should) comemnt this line if you can still use cuda 

    predictor = DefaultPredictor(cfg) #run predictor
    outputs = predictor(im)
    # now I visualize the instances'segmentation (also from the original version)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2) #this "scale" needs to be tuned to fit the photo
    # if the photo is so small, you may want the scale to be bigger and vice versa
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])  # show instances segmentation