import torch
import cv2
import numpy as np
import supervision as sv
from mmocr.utils.polygon_utils import poly2bbox
from mmocr.apis.inferencers import MMOCRInferencer

#set the device

DEVICE =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print (DEVICE)

detection_config_path=r'mmocr\configs\textdet\dbnetpp\dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.py'
detection_weight_path=r'mmocr\weights\DBNet\dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac (1).pth'
recognition_config_path=r'mmocr\configs\textrecog\abinet\abinet-vision_20e_st-an_mj.py'
recognition_weight_path=r'mmocr\weights\ABINet\abinet_20e_st-an_mj_20221005_012617-ead8c139 (1).pth'

mmocr_inference = MMOCRInferencer(
    det=detection_config_path,
    det_weights=detection_weight_path,
    rec=recognition_config_path,
    rec_weights=recognition_weight_path,
    device=DEVICE
)

path=r'D:\javeed\project\16.jpg'

image_bgr=cv2.imread(path)
original_image=image_bgr
#convert to rgb
image_rgb=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
#resize
image_rgb=cv2.resize(image_rgb,(1024,1024))
image_bgr=cv2.resize(image_bgr,(1024,1024))

result=mmocr_inference(image_rgb)

# print(result.keys())

result=mmocr_inference(image_rgb)['predictions'][0]
recognized_text=result['rec_texts']
detected_polygons=result['det_polygons']

print(detected_polygons)
print(recognized_text)

detected_boxes=torch.tensor(
    np.array([poly2bbox(poly) for poly in detected_polygons])
)

detected_boxes=np.array(detected_boxes)
print( detected_boxes)

counter=0
for bbox in detected_boxes:
  x1,y1, x2, y2=bbox
  image_bgr=cv2.rectangle(image_bgr, (int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3)
  image_bgr=cv2.putText(image_bgr,recognized_text[counter],(int(x1),int(y1)),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,255),2)
  counter+=1

sv.plot_images_grid (
images=[original_image, image_bgr],
grid_size=(1,2),
titles=['Original image', 'MMOCR Image']
)

cv2.imwrite('abi_dbnet1.jpeg', image_bgr)