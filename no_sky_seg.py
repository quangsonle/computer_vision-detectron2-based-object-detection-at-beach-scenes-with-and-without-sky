
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.9")   # please manually install torch 1.9 if Colab changes its default version
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow
import cv2 as cv
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
def hough_sky_seg(im):
    gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150,apertureSize = 3)
    lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=200,maxLineGap=10)
    y_bor=im.shape[1]
    for line in lines:
        x1,y1,x2,y2 = line[0]
        #cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        if (y1<y_bor): 
         if(y2<y1):
          y_bor=y2
         else:
          y_bor=y1
    im=im[y_bor:,:]
    cfg = get_cfg()
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
 