'''
    戴口罩检测Demo
    cam_id = 1 为手机前置摄像头
    cam_id = 0 为手机后置摄像头
'''
import time
import aidlite_gpu
import numpy as np
from cvs import cvs, cv2
from qiniu import Auth, put_file
from qiniu import BucketManager
from qiniu import CdnManager
from utils import preprocess_img, convert_shape, \
                  single_class_non_max_suppression, \
                  decode_bbox, generate_anchors, draw_result



# 配置七牛云信息,需要替换一下，这里我瞎写的
access_key = "JGBgHa_7121aKIGv7a_8azuAa-gvkgN2Ci1a5XrNE0D"
secret_key = "SeDybkKBMs222_HoBHUkk8ElhdZ6DWM_X1CwwUoVEWR"
bucket_name = "suook1212123"
bucket_url = "rgu8zgw2x.hbffff-bkt.clouddn.com"
q = Auth(access_key, secret_key)
bucket = BucketManager(q)
cdnManager = CdnManager(q)

# 将本地图片上传到七牛云中
def upload_img(bucket_name, file_name, file_path):
    # generate token
    token = q.upload_token(bucket_name, file_name, 3600)
    put_file(token, file_name, file_path)

# 获得七牛云服务器上file_name的图片外链
def get_img_url(bucket_url, file_name):
    img_url = 'http://%s/%s' % (bucket_url, file_name)
    return img_url



# 定义输入输出shape
input_shape = (1, 260, 260, 3) #image_shape
output_shape = [(1, 5972, 4), (1, 5972, 2)] #loc_shape, cls_shape

# 定义一些参数
model_path = "./face_mask_detection.tflite"
id2class = {0: 'Mask', 1: 'NoMask'}
anchors_exp = np.expand_dims(generate_anchors(), 0)
result_format = "Avg_FPS:{:^4.2f}, OnTime_FPS:{:^4.2f}."

# 载入模型
aidlite = aidlite_gpu.aidlite()
# 载入口罩模型
aidlite.ANNModel(model_path, 
                 convert_shape(input_shape, False), 
                 convert_shape(output_shape, False),
                 4, 
                 0)

# 获得前置摄像头
cam_id = 1
cap = cvs.VideoCapture(cam_id)
cost_time = 0.1
while True:
    image = cap.read()
    if image is None:
        time.sleep(0.5)
        continue
    if 1 == cam_id:
        # 前置摄像头就水平翻转一下
        image = cv2.flip(image, 1)
    
    # 推理
    t0 = time.time()
    aidlite.setTensor_Fp32(preprocess_img(image, 
                                          target_shape=(input_shape[1], input_shape[2]),
                                          div_num=255,
                                          means=None,
                                          stds=None))
    aidlite.invoke()
    loc_res = aidlite.getTensor_Fp32(0).reshape(*output_shape[0])
    cls_res = aidlite.getTensor_Fp32(1).reshape(*output_shape[1])[0]
    
    # 后处理
    y_bboxes = decode_bbox(anchors_exp, loc_res)[0]
    bbox_max_scores = np.max(cls_res, axis=1)
    bbox_max_score_classes = np.argmax(cls_res, axis=1)
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=0.6,
                                                 iou_thresh=0.4,
                                                 )
    # 画出来
    image = draw_result(image, keep_idxs, y_bboxes, bbox_max_scores, bbox_max_score_classes, id2class)
    t1 = time.time()

    cost_time = 0.8*cost_time + 0.2*(t1-t0)
    cvs.setLbs(result_format.format(1/cost_time, 1/(t1-t0)))


    image_up_name = image
    # 上传到七牛云后，保存成的图片名称
    image_qiniu_name = "mask_error.jpg"
    upload_img(bucket_name, image_qiniu_name, image_up_name)

    url_receive = get_img_url(bucket_url, image_qiniu_name)
    refresh = cdnManager.refresh_urls([url_receive])


    cvs.imshow(image)
