# objects detection at beach scenes usig detectron-sky detection and sky removal using panoptic segmentation (detectron) and hough line transform

this is a work to demo the usage of open source detectron to detect and label objects at beach scenes (there are 4 photos)

the performance is evaluated if direct applying detectron2, applying detectron2 after sky removal (i.e no object on the sky in the beach scenes) with 2 different ways to detect the sky:panoptic segmentation (detectron2) and hough line transform (opencv)

it is run on Google Colab. Running on offiline GPU should work but is not guaranteed if there is any issue

in order to run:
1/ i shared the link of my google colab with you (any body can access it via the link)

2/ after accessing my notebook on google colab, upload 4 photos and 3 independent python files:
-beach.jpg, beach2.jpg, beach3.jpg, beach4.jpg
-sky_crop.py,no_sky_seg.py,alone_seg.py
they all are inside of the the project's page
ENSURE you put all of them in "content" folder on the browse, otherwise remote machine wont find them and of course, nothing will work
3/ you can run one-by-one of the box to check


the link of my google colab:

https://colab.research.google.com/drive/1WyWEpP7dg5MjmJagJ3ErcpD-XLg6VNeQ?usp=sharing
