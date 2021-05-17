import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from skimage import measure

# ------------------------------获得图片网格和角点图
def getPointGrid(src_img, distance_threshold, h_size, v_size, extendFlag=False):
    if len(src_img.shape) == 2:  # 灰度图
        gray_img = src_img
    else:
        gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    thresh_img = cv2.adaptiveThreshold(~gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    h_img = thresh_img.copy()
    v_img = thresh_img.copy()

    # ------------------------------形态学提取横竖线
    h_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    h_erode_img = cv2.erode(h_img, h_structure, 1)
    h_dilate_img = cv2.dilate(h_erode_img, h_structure, 1)

    v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
    v_erode_img = cv2.erode(v_img, v_structure, 1)
    v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)
    
    mask_img = h_dilate_img+v_dilate_img

    #------------------------------延长横竖线，增加新角点
    if extendFlag:
        h_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (distance_threshold, 1))
        h_dilate_img = cv2.dilate(h_dilate_img, h_dilate_kernel, anchor=(distance_threshold//2, 0))
        v_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, distance_threshold))
        v_dilate_img = cv2.dilate(v_dilate_img, v_dilate_kernel, anchor=(0, distance_threshold//2))
    
    joints_img = cv2.bitwise_and(h_dilate_img,v_dilate_img)
   
    return mask_img, joints_img 

#------------------------------提取角点中心坐标,并除去折痕附近的角点
def getCenter(mask, new_joint, distance_threshold, creaseFlag=False):
    new_mask = mask.copy()
    point_threshold = 10
    # ------------------------------聚集角点
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (distance_threshold, distance_threshold))
    new_joint = cv2.dilate(new_joint, dilate_kernel, iterations=1)
    new_joint = cv2.erode(new_joint, dilate_kernel, iterations=1)
    # ------------------------------获得中心位置
    label_image = measure.label(new_joint, connectivity=2)
    label_att = measure.regionprops(label_image)
    pointcenter = [(int(point.centroid[0]), int(point.centroid[1])) for point in label_att]
    if not creaseFlag:
        return pointcenter
    else:
        # ------------------------------获得表格区域
        h_start, h_end = new_mask.shape[0], point_threshold
        w_start, w_end = new_mask.shape[1], point_threshold
        for h, w in pointcenter:
            if point_threshold < h < new_mask.shape[0]-point_threshold:
                h_start = min(h_start, h)
                h_end = max(h_end, h)
            if point_threshold < w < new_mask.shape[1]-point_threshold:
                w_start = min(w_start, w)
                w_end = max(w_end, w)
        h_start, h_end = h_start - point_threshold, h_end + point_threshold
        w_start, w_end = w_start - point_threshold, w_end + point_threshold
        # ------------------------------增强mask
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (distance_threshold, distance_threshold))
        new_mask = cv2.dilate(new_mask, dilate_kernel, iterations=1)
        new_mask = cv2.erode(new_mask, dilate_kernel, iterations=1)  
        # ------------------------------直线检测
        lines = cv2.HoughLinesP(new_mask, rho=1, theta=np.pi / 360, threshold=100, minLineLength=100, maxLineGap=distance_threshold)
        crease = []# ------------------------------折痕提取
        try:
            for line in lines:
                w1, h1, w2, h2 = line[0]
                if h_start < h1 < h_end and h_start < h2 < h_end and w_start < w1 < w_end and w_start < w2 < w_end:
                    continue
                if abs(h1-h2)<abs(w1-w2) and h1<h_start+2*point_threshold and h2<h_start+2*point_threshold:
                    continue
                if abs(h1-h2)<abs(w1-w2) and h1>h_end-2*point_threshold and h2>h_end-2*point_threshold:
                    continue
                if abs(h1-h2)>abs(w1-w2) and w1<w_start+2*point_threshold and w2<w_start+2*point_threshold:
                    continue
                if abs(h1-h2)>abs(w1-w2) and w1>w_end-2*point_threshold and w2>w_end-2*point_threshold:
                    continue
                A, B, C = w1 - w2, h2 - h1, h1 * w2 - h2 * w1
                crease.append((A, B, C))
        except:
            pass
        # ------------------------------去除距离折痕过近的点
        points = []
        for x, y in pointcenter:
            distance = point_threshold / 2
            for A, B, C in crease:
                distance = min(distance, abs(A * x + B * y + C) / (A ** 2 + B ** 2) ** 0.5)
            if distance >= point_threshold / 2:
                points.append((x, y))

        return points

#------------------------------生成角点map
def pointMap(points):
    heightKeys = []
    heightMap = {}
    widthKeys = []
    widthMap = {}
    for p in points:
        if p[0] in heightMap:
            heightMap[p[0]].append(p[1])
        else:
            heightMap[p[0]] = [p[1]]
            heightKeys.append(p[0])
        if p[1] in widthMap:
            widthMap[p[1]].append(p[0])
        else:
            widthMap[p[1]] = [p[0]]
            widthKeys.append(p[1])
    heightKeys.sort()
    widthKeys.sort()
    for k in heightKeys:
        heightMap[k].sort()
    for k in widthKeys:
        widthMap[k].sort()

    return heightMap, heightKeys, widthMap, widthKeys

#------------------------------判断两点连接与否
def isConnected(point1, point2, mask, direction):
    if direction == 'v':
        w = (point1[1] + point2[1]) // 2
        w1, w2 = max(w-5,0), min(w+6,mask.shape[1])
        h_min = min(point1[0], point2[0])
        h_max = max(point1[0], point2[0])
        return (np.max(mask[h_min:h_max+1,w1:w2], axis=1) > 100).sum() >= (h_max-h_min) * 0.85

    if direction == 'h':
        h = (point1[0] + point2[0]) // 2
        h1, h2 = max(h-5,0), min(h+6,mask.shape[0])
        w_min = min(point1[1], point2[1])
        w_max = max(point1[1], point2[1])
        return (np.max(mask[h1:h2,w_min:w_max+1], axis=0) > 100).sum() >= (w_max-w_min) * 0.85

#------------------------------寻找右边连接的点
def findRightPoint(point, widthMap, widthKeys, mask, distance_threshold):
    ans = []
    for w in widthKeys:
        if w - point[1] < distance_threshold:
            continue
        for h in widthMap[w]:
            if abs(h - point[0]) < distance_threshold:
                if isConnected(point, (h, w), mask, 'h'):
                    ans.append((h, w))
                else:
                    return ans
    return ans

#------------------------------寻找下方连接的点
def findBottomPoint(point, heightMap, heightKeys, mask, distance_threshold):
    ans = []
    for h in heightKeys:
        if h - point[0] < distance_threshold:
            continue
        for w in heightMap[h]:
            if abs(w - point[1]) < distance_threshold:
                if isConnected(point, (h, w), mask, 'v'):
                    ans.append((h, w))
                else:
                    return ans
    return ans

#------------------------------寻找最后一个点，构成矩形
def findFinalPoint(bottomLeft, topRight, heightMap, heightKeys, mask, distance_threshold):
    for bl in bottomLeft:
        for tr in topRight:
            h, w = bl[0], tr[1]
            for end_h in heightKeys:
                if abs(end_h - h) > distance_threshold:
                    continue
                for end_w in heightMap[end_h]:
                    if abs(end_w - w) > distance_threshold:
                        continue
                    if isConnected(bl, (end_h, end_w), mask, 'h') and isConnected(tr, (end_h, end_w), mask, 'v'):
                        return True, (end_h, end_w)
    return False, (0, 0)

#------------------------------寻找矩形
def searchCell(heightMap, heightKeys, widthMap, widthKeys, mask, distance_threshold, thre_h, thre_w):
    rects = []
    for start_h in heightMap:
        for start_w in heightMap[start_h]:
            point = (start_h, start_w)
            topRight = findRightPoint(point, widthMap, widthKeys, mask, distance_threshold)
            bottomLeft = findBottomPoint(point, heightMap, heightKeys, mask, distance_threshold)
            flag, finalPoint = findFinalPoint(bottomLeft, topRight, heightMap, heightKeys, mask, distance_threshold)
            if flag:
                rects.append((start_h, start_w, finalPoint[0], finalPoint[1]))
    return rects

#------------------------------去除过小的框
def remove_noise(rects, thre_h, thre_w):
    need_rects = []
    for rect in rects:
        if (rect[2] - rect[0]) < thre_h and (rect[3] - rect[1]) < thre_w:
            continue
        need_rects.append(rect)
    return need_rects

def init_filepath(outpath):
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.makedirs(outpath)

def is_have_table(path):
    marg = 2
    #距离阈值，用于判断两个点是否相连，以及横竖线延长
    distance_threshold = 12
    #形态学变量，表示水平/竖直方向的像素数量，小于阈值的直线会被滤除，不加入mask
    h_size, v_size = 30, 30
    #高/宽阈值，尺寸小于阈值的矩形最后会被剔除
    thre_h, thre_w = 30, 30
    #延长横竖线，获得额外角点
    extendFlag = True
    #去除折痕影响
    creaseFlag = True

    img_path, outpath = path[0], path[1]
    if outpath:
        init_filepath(outpath)
    
    img = cv2.imread(img_path)
    
    half_img = cv2.pyrDown(img)# 下采样

    #获取网格mask和角点图
    new_mask, new_joint = getPointGrid(half_img, distance_threshold, h_size, v_size, extendFlag=extendFlag)

    #聚集角点，获取角点的中心坐标，去除折痕附近角点
    averagedNodes = getCenter(new_mask, new_joint, distance_threshold, creaseFlag=creaseFlag)

    #把角点存储为字典，用于后续查询
    heightMap, heightKeys, widthMap, widthKeys = pointMap(averagedNodes)    

    #查找矩形框
    rects = searchCell(heightMap, heightKeys, widthMap, widthKeys, new_mask, distance_threshold, thre_h, thre_w)

    #剔除过小尺寸框
    rects = remove_noise(rects, thre_h, thre_w)
       
    ishave, cell = False, []
    if len(rects) >= 4:
        ishave = True
    for rect in rects:
        h_start, w_start, h_end, w_end = rect[0]*2+marg, rect[1]*2+marg, rect[2]*2-marg, rect[3]*2-marg
        if h_start >= h_end or w_start >= w_end or h_end < 0 or w_end < 0:
            h_start, w_start, h_end, w_end = rect[0] * 2, rect[1] * 2, rect[2] * 2, rect[3] * 2
        if h_start >= h_end or w_start >= w_end or h_end < 0 or w_end < 0: continue
        crop_img = img[h_start:h_end, w_start:w_end]
        pos_flag = str(h_start) + '-' + str(w_start) + '_' + str(h_end) + '-' + str(w_end)
        cell.append(pos_flag)
        if outpath:
            cv2.imwrite(os.path.join(outpath, pos_flag + ".jpg"), crop_img)
  
    return ishave, cell

def showCell(path,extendFlag = True,creaseFlag = True):
    #距离阈值，用于判断两个点是否相连，以及横竖线延长
    distance_threshold = 12

    #形态学变量，表示水平/竖直方向的像素数量，小于阈值的直线会被滤除，不加入mask
    h_size, v_size = 30, 30

    #高/宽阈值，尺寸小于阈值的矩形最后会被剔除
    thre_h, thre_w = 30, 30
    
#     #延长横竖线，获得额外角点
#     extendFlag = True
#     #去除折痕影响
#     creaseFlag = True
    
    img = cv2.imread(path)
    
    half_img = cv2.pyrDown(img)# 下采样

    #获取网格mask和角点图
    new_mask, new_joint = getPointGrid(half_img, distance_threshold, h_size, v_size, extendFlag=extendFlag)

    #聚集角点，获取角点的中心坐标，去除折痕附近角点
    averagedNodes = getCenter(new_mask, new_joint, distance_threshold, creaseFlag=creaseFlag)

    #把角点存储为字典，用于后续查询
    heightMap, heightKeys, widthMap, widthKeys = pointMap(averagedNodes)    

    #查找矩形框
    rects = searchCell(heightMap, heightKeys, widthMap, widthKeys, new_mask, distance_threshold, thre_h, thre_w)

    #剔除过小尺寸框
    rects = remove_noise(rects, thre_h, thre_w)
       
    ishave, cell = False, []
    if len(rects) >= 4:
        ishave = True
        for rect in rects:
            h_start, w_start, h_end, w_end = rect[0]*2, rect[1]*2, rect[2]*2, rect[3]*2
            if h_start >= h_end or w_start >= w_end or h_end < 0 or w_end < 0: continue
            cell.append([(w_start, h_start), (w_end, h_end)])
  
    return ishave, cell

def showTable(img_path,extendFlag = True,creaseFlag = True):
    distance_threshold = 12
    thre_h, thre_w = 30, 30
    h_size, v_size = 30, 30
#     #延长横竖线，获得额外角点
#     extendFlag = True
#     #去除折痕影响
#     creaseFlag = True
    
    img = cv2.imread(img_path)

    half_img = cv2.pyrDown(img)  # 下采样

    new_mask, new_joint = getPointGrid(half_img, distance_threshold, h_size, v_size, extendFlag=extendFlag)
                       

    #聚集角点，获取角点的中心坐标，去除折痕附近角点
    averagedNodes = getCenter(new_mask, new_joint, distance_threshold, creaseFlag=creaseFlag)
    
    heightMap, heightKeys, widthMap, widthKeys = pointMap(averagedNodes)

    rects = searchCell(heightMap, heightKeys, widthMap, widthKeys, new_mask, distance_threshold, thre_h, thre_w)

    rects = remove_noise(rects, 30, 30)

    ishave, cell = False, []
    if len(rects) >= 4:
        ishave = True
        for rect in rects:
            h_start, w_start, h_end, w_end = rect[0] * 2, rect[1] * 2, rect[2] * 2, rect[3] * 2
            if h_start >= h_end or w_start >= w_end or h_end < 0 or w_end < 0: continue
            cv2.rectangle(img, (w_start, h_start), (w_end, h_end), (255, 0, 0), 2)
            cv2.circle(img, ((w_start + w_end) // 2, (h_start + h_end) // 2), 5, (0, 0, 255), 3)
    return img

def showMask(img,extendFlag = True,creaseFlag = False):
    distance_threshold = 12
    thre_h, thre_w = 30, 30
    h_size, v_size = 30, 30

    new_mask, new_joint = getPointGrid(img, distance_threshold, h_size, v_size, extendFlag=extendFlag)

    #聚集角点，获取角点的中心坐标，去除折痕附近角点
    averagedNodes = getCenter(new_mask, new_joint, distance_threshold, creaseFlag=creaseFlag)

    heightMap, heightKeys, widthMap, widthKeys = pointMap(averagedNodes)

    rects = searchCell(heightMap, heightKeys, widthMap, widthKeys, new_mask, distance_threshold, thre_h, thre_w)
    

    rects = remove_noise(rects, 60, 60)
    
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    new_mask = cv2.dilate(new_mask, dilate_kernel, iterations=1)
    new_joint = cv2.dilate(new_joint, dilate_kernel, iterations=1)
                                       
    mask_img = np.zeros(new_mask.shape, np.uint8)
    joint_img = np.zeros(new_joint.shape, np.uint8)
    for h_start, w_start, h_end, w_end in rects:
        if h_start >= h_end or w_start >= w_end or h_end < 0 or w_end < 0: continue
        cv2.rectangle(mask_img, (w_start, h_start), (w_end, h_end), (255, 255, 255), 7)
        for x, y in [(w_start,h_start),(w_end,h_start),(w_start,h_end),(w_end,h_end)]:
            cv2.circle(joint_img, (x, y), 8, (255,255,255), -1)
    mask_img = cv2.adaptiveThreshold(mask_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)    
    
    structure = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    
    mask_dilate_img = cv2.dilate(mask_img, structure, 1)
    mask_erode_img = cv2.erode(mask_dilate_img, structure, 1)
    joint_dilate_img = cv2.dilate(joint_img, structure, 1)
    joint_erode_img = cv2.erode(joint_dilate_img, structure, 1)
    
    
    return cv2.bitwise_and(mask_erode_img, new_mask), cv2.bitwise_and(joint_erode_img, new_joint)

def get_table_pos(img_path, extendFlag = True, creaseFlag = False):
    marg = 2
    #距离阈值，用于判断两个点是否相连，以及横竖线延长
    distance_threshold = 12
    #形态学变量，表示水平/竖直方向的像素数量，小于阈值的直线会被滤除，不加入mask
    h_size, v_size = 30, 30
    #高/宽阈值，尺寸小于阈值的矩形最后会被剔除
    thre_h, thre_w = 30, 30
    #延长横竖线，获得额外角点
    
    img = cv2.imread(img_path)
    
    half_img = cv2.pyrDown(img)# 下采样

    #获取网格mask和角点图
    new_mask, new_joint = getPointGrid(half_img, distance_threshold, h_size, v_size, extendFlag=extendFlag)

    #聚集角点，获取角点的中心坐标，去除折痕附近角点
    averagedNodes = getCenter(new_mask, new_joint, distance_threshold, creaseFlag=creaseFlag)

    #把角点存储为字典，用于后续查询
    heightMap, heightKeys, widthMap, widthKeys = pointMap(averagedNodes)    

    #查找矩形框
    rects = searchCell(heightMap, heightKeys, widthMap, widthKeys, new_mask, distance_threshold, thre_h, thre_w)

    #剔除过小尺寸框
    rects = remove_noise(rects, thre_h, thre_w)
       
    cell = []
    for rect in rects:
        h_start, w_start, h_end, w_end = rect[0]*2+marg, rect[1]*2+marg, rect[2]*2-marg, rect[3]*2-marg
        if h_start >= h_end or w_start >= w_end or h_end < 0 or w_end < 0:
            h_start, w_start, h_end, w_end = rect[0] * 2, rect[1] * 2, rect[2] * 2, rect[3] * 2
        if h_start >= h_end or w_start >= w_end or h_end < 0 or w_end < 0: continue
        pos_flag = str(h_start) + '-' + str(w_start) + '_' + str(h_end) + '-' + str(w_end)
        cell.append(pos_flag)
  
    return cell


if __name__ == '__main__':
    pass