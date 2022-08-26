import cv2
import numpy as np

def preprocess_img(img, target_shape:tuple=None, div_num=255, means:list=[0.485, 0.456, 0.406], stds:list=[0.229, 0.224, 0.225]):
    '''
    图像预处理:
    target_shape: 目标shape
    div_num: 归一化除数
    means: len(means)==图像通道数，通道均值, None不进行zscore
    stds: len(stds)==图像通道数，通道方差, None不进行zscore
    '''
    img_processed = np.copy(img)
    # resize
    if target_shape:
        img_processed = cv2.resize(img_processed, target_shape)

    img_processed = img_processed.astype(np.float32)
    img_processed = img_processed/div_num

    # z-score
    if means is not None and stds is not None:
        means = np.array(means).reshape(1, 1, -1)
        stds = np.array(stds).reshape(1, 1, -1)
        img_processed = (img_processed-means)/stds

    # unsqueeze
    img_processed = img_processed[None, :]

    return img_processed.astype(np.float32)

def convert_shape(shapes:tuple or list, int8=False):
    '''
    转化为aidlite需要的格式
    '''
    if isinstance(shapes, tuple):
        shapes = [shapes]
    out = []
    for shape in shapes:
        nums = 1 if int8 else 4
        for n in shape:
            nums *= n
        out.append(nums)
    return out

def single_class_non_max_suppression(bboxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):
    '''
    单类NMS算法
    Hint: for the specific class, given the bbox and its confidence,
    1) sort the bbox according to the confidence from top to down, we call this a set
    2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
    3) remove the bbox whose IOU is higher than the iou_thresh from the set,
    4) loop step 2 and 3, util the set is empty.
    :param bboxes: numpy array of 2D, [num_bboxes, 4]
    :param confidences: numpy array of 1D. [num_bboxes]
    :param conf_thresh:
    :param iou_thresh:
    :param keep_top_k:
    :return:
    '''
    if len(bboxes) == 0: return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]

    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    idxs = np.argsort(confidences)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # keep top k
        if keep_top_k != -1:
            if len(pick) >= keep_top_k:
                break

        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idxs = np.delete(idxs, need_to_be_deleted_idx)

    return conf_keep_idx[pick]

def decode_bbox(anchors, raw_outputs, variances=[0.1, 0.1, 0.2, 0.2]):
    '''
    结合anchor来解算bbox
    Decode the actual bbox according to the anchors.
    the anchor value order is:[xmin,ymin, xmax, ymax]
    :param anchors: numpy array with shape [batch, num_anchors, 4]
    :param raw_outputs: numpy array with the same shape with anchors
    :param variances: list of float, default=[0.1, 0.1, 0.2, 0.2]
    :return:
    '''
    anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
    anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
    anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
    anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]
    raw_outputs_rescale = raw_outputs * np.array(variances)
    predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
    predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
    predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
    predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
    predict_xmin = predict_center_x - predict_w / 2
    predict_ymin = predict_center_y - predict_h / 2
    predict_xmax = predict_center_x + predict_w / 2
    predict_ymax = predict_center_y + predict_h / 2
    predict_bbox = np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)
    return predict_bbox

def generate_anchors(feature_map_sizes=[[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]], 
                     anchor_sizes=[[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]], 
                     anchor_ratios=[[1, 0.62, 0.42]] * 5, 
                     offset=0.5):
    '''
    产生anchors
    :param feature_map_sizes: list of list, for example: [[40,40], [20,20]]
    :param anchor_sizes: list of list, for example: [[0.05, 0.075], [0.1, 0.15]]
    :param anchor_ratios: list of list, for example: [[1, 0.5], [1, 0.5]]
    :param offset: default to 0.5
    :return:
    '''
    anchor_bboxes = []
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
        cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
        cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
        center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

        num_anchors = len(anchor_sizes[idx]) +  len(anchor_ratios[idx]) - 1
        center_tiled = np.tile(center, (1, 1, 2* num_anchors))
        anchor_width_heights = []

        # different scales with the first aspect ratio
        for scale in anchor_sizes[idx]:
            ratio = anchor_ratios[idx][0] # select the first ratio
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        # the first scale, with different aspect ratios (except the first one)
        for ratio in anchor_ratios[idx][1:]:
            s1 = anchor_sizes[idx][0] # select the first scale
            width = s1 * np.sqrt(ratio)
            height = s1 / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        bbox_coords = center_tiled + np.array(anchor_width_heights)
        bbox_coords_reshape = bbox_coords.reshape((-1, 4))
        anchor_bboxes.append(bbox_coords_reshape)
    anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
    return anchor_bboxes

def draw_result(image, keep_idxs, y_bboxes, bbox_max_scores, bbox_max_score_classes, id2class):
    '''
    把检测结果画出来
    '''
    height, width, _ = image.shape
    for idx in keep_idxs:
        conf = bbox_max_scores[idx]
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if class_id == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, "%s: %2.2f" % (id2class[class_id], conf), (xmin + 2, ymin + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color)
    return image