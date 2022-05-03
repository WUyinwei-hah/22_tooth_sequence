import os
import cv2
import paddlex as pdx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')


def get_color_map_list(num_classes):
    """ Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
    Args:
        num_classes: Number of classes
    Returns:
        The color map
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def draw_bbox_mask(image, results, threshold=0.5, color_map=None):
    _SMALL_OBJECT_AREA_THRESH = 1000
    height, width = image.shape[:2]
    default_font_scale = max(np.sqrt(height * width) // 900, .5)
    linewidth = max(default_font_scale / 40, 2)

    labels = list()
    for dt in results:
        if dt['category'] not in labels:
            labels.append(dt['category'])

    if color_map is None:
        color_map = get_color_map_list(len(labels) + 2)[2:]
    else:
        color_map = np.asarray(color_map)
        if color_map.shape[0] != len(labels) or color_map.shape[1] != 3:
            raise Exception(
                "The shape for color_map is required to be {}x3, but recieved shape is {}x{}.".
                    format(len(labels), color_map.shape))
        if np.max(color_map) > 255 or np.min(color_map) < 0:
            raise ValueError(
                " The values in color_map should be within 0-255 range.")

    keep_results = []
    areas = []
    for dt in results:

        cname, bbox, score = dt['category'], dt['bbox'], dt['score']
        label = dt.get("pos", "unknown")
        if score < threshold:
            continue
        keep_results.append(dt)
        areas.append(bbox[2] * bbox[3])
    areas = np.asarray(areas)
    sorted_idxs = np.argsort(-areas).tolist()
    keep_results = [keep_results[k]
                    for k in sorted_idxs] if keep_results else []

    for dt in keep_results:
        cname, bbox, score = dt['category'], dt['bbox'], dt['score']
        label = dt.get("pos", "unknown")
        bbox = list(map(int, bbox))
        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h

        color = tuple(map(int, color_map[labels.index(cname)]))
        # draw bbox
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color,
                              linewidth)

        # draw mask
        if 'mask' in dt:
            mask = dt['mask'] * 255
            image = image.astype('float32')
            alpha = .7
            w_ratio = .4
            color_mask = np.asarray(color, dtype=int)
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
            idx = np.nonzero(mask)
            image[idx[0], idx[1], :] *= 1.0 - alpha
            image[idx[0], idx[1], :] += alpha * color_mask
            image = image.astype("uint8")
            contours = cv2.findContours(
                mask.astype("uint8"), cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_NONE)[-2]
            image = cv2.drawContours(
                image,
                contours,
                contourIdx=-1,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA)

        # draw label
        text_pos = (xmin, ymin)
        instance_area = w * h
        if (instance_area < _SMALL_OBJECT_AREA_THRESH or h < 40):
            if ymin >= height - 5:
                text_pos = (xmin, ymin)
            else:
                text_pos = (xmin, ymax)
        height_ratio = h / np.sqrt(height * width)
        font_scale = (np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2,
                              2) * 0.5 * default_font_scale)
        # text = "{} {:.2f}".format(cname, score)
        text = "{} {}".format(cname, label)
        (tw, th), baseline = cv2.getTextSize(
            text,
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=font_scale,
            thickness=1)
        image = cv2.rectangle(
            image,
            text_pos, (text_pos[0] + tw, text_pos[1] + th + baseline),
            color=color,
            thickness=-1)
        image = cv2.putText(
            image,
            text, (text_pos[0], text_pos[1] + th),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA)

    return image


def calc_iou(x1, y1, x2, y2, a1, b1, a2, b2):
    """
    计算iou
    :param x1: bbox1的左上角x
    :param y1: bbox1的左上角y
    :param x2: bbox1的右下角x
    :param y2: bbox1的右下角y
    :param a1: bbox2的左上角x
    :param b1: bbox2的左上角y
    :param a2: bbox2的右下角x
    :param b2: bbox2的右下角y
    :return:
    """

    ax = max(x1, a1)  # 相交区域左上角横坐标
    ay = max(y1, b1)  # 相交区域左上角纵坐标
    bx = min(x2, a2)  # 相交区域右下角横坐标
    by = min(y2, b2)  # 相交区域右下角纵坐标

    area_N = (x2 - x1) * (y2 - y1)
    area_M = (a2 - a1) * (b2 - b1)

    w = max(0, bx - ax)
    h = max(0, by - ay)
    area_X = w * h

    return area_X / (area_N + area_M - area_X)


def nms(bbox_array, total_tooth_count):
    """
    做非极大抑制
    :param bbox_array: bbox数组，已经按照从大大到小排序
    :return: 完成非极大抑制后的数组，长度小于等于total_tooth_count
    """

    # 用来存放最后结果的数组
    result_array = []
    # 当剩余的未筛选的bbox数量大于0并且结果数量没超过要求
    while len(bbox_array) > 0 and len(result_array) < total_tooth_count:
        # 将置信度最高的放入
        best_bbox = bbox_array[0]
        result_array.append(best_bbox)
        # 将已经放入的删除
        del bbox_array[0]
        # 计算其他bbox与置信度最高的bbox的iou
        best_bbox_rect = list(map(int, best_bbox["bbox"]))
        index_to_del = []
        for index, bbox_item in enumerate(bbox_array):
            bbox_item_rect = list(map(int, bbox_item["bbox"]))
            if calc_iou(best_bbox_rect[0], best_bbox_rect[1],
                        best_bbox_rect[0] + best_bbox_rect[2], best_bbox_rect[1] + best_bbox_rect[3],
                        bbox_item_rect[0], bbox_item_rect[1],
                        bbox_item_rect[0] + bbox_item_rect[2], bbox_item_rect[1] + bbox_item_rect[3]
                        ) > 0.3:
                # 将大于阈值的加入待删除列表
                index_to_del.insert(0, index)
        # 删除
        for index in index_to_del:
            del bbox_array[index]

    return result_array


def filter_pip_line(result, type):
    """
    经过一系列过滤，得到最后的预测结果result
    :param result: 预测结果，是一个bbox数组
    :return: 经过处理后的预测结果，是一个数组
    """
    type_map = {
        "front": [1, 2, 3],
        "side": [3, 4, 5, 6],
        "inside": [1, 2, 3, 4, 5, 6, 7, 8],
    }

    tooth_dict = {
    }

    # 正面4颗， 侧面和口内2颗
    if type == "front":
        total_tooth_count = 4
    else:
        total_tooth_count = 2

    for label in type_map[type]:
        tooth_dict[str(label)] = []

    # 1.对1 - 3 进行分类
    for bbox in result:
        if tooth_dict.get(bbox["category"]) is not None:
            tooth_dict[bbox["category"]].append(bbox)
    # 2.每个分类挑选前四颗置信度最高的
    for item in tooth_dict.items():
        label = item[0]
        bbox_array = item[1]
        # 过滤置信度过小的
        bbox_array = list(filter(lambda bbox: bbox["score"] > 0.50, bbox_array))
        # 从大到小排列
        bbox_array = sorted(bbox_array, key=lambda bbox: -bbox["score"])

        # 3.如果前total_tooth_count颗中有IOU过大（上下是可能有交集）的，剔除置信度小的牙齿，再添加本分类的一颗牙齿，如循环到没有牙齿或者符合条件
        tooth_dict[label] = nms(bbox_array, total_tooth_count)

    # 4. 根据坐标xy从大大小排序
    for new_bbox_array in tooth_dict.values():
        # 正面
        # （大，大）左下（大，小）左上（小，大）右下（小，小）右上
        if type == "front":
            if len(new_bbox_array) > 1:
                # 计算坐标中心
                x_sum = 0
                y_sum = 0
                for bbox in new_bbox_array:
                    x_sum += bbox["bbox"][0] + bbox["bbox"][2] / 2
                    y_sum += bbox["bbox"][1] + bbox["bbox"][3] / 2
                x_avg = x_sum / len(new_bbox_array)
                y_avg = y_sum / len(new_bbox_array)

                # 贴标签
                for bbox in new_bbox_array:
                    center_x = bbox["bbox"][0] + bbox["bbox"][2] / 2
                    center_y = bbox["bbox"][1] + bbox["bbox"][3] / 2
                    if center_x >= x_avg and center_y >= y_avg:
                        bbox["pos"] = "ld"
                    elif center_x < x_avg and center_y >= y_avg:
                        bbox["pos"] = "rd"
                    elif center_x < x_avg and center_y < y_avg:
                        bbox["pos"] = "ru"
                    else:
                        bbox["pos"] = "lu"

            elif len(new_bbox_array) < 3:
                # 等于1或者2的情况，正面根据1
                standard_tooth_array = tooth_dict["1"]
                # 正面是4颗
                if len(standard_tooth_array) == 4:
                    x_sum = 0
                    y_sum = 0
                    for bbox in standard_tooth_array:
                        x_sum += bbox["bbox"][0] + bbox["bbox"][2] / 2
                        y_sum += bbox["bbox"][1] + bbox["bbox"][3] / 2
                    x_avg = x_sum / len(standard_tooth_array)
                    y_avg = y_sum / len(standard_tooth_array)

                    for bbox in new_bbox_array:
                        center_x = bbox["bbox"][0] + bbox["bbox"][2] / 2
                        center_y = bbox["bbox"][1] + bbox["bbox"][3] / 2
                        if center_x >= x_avg and center_y >= y_avg:
                            bbox["pos"] = "ld"
                        elif center_x < x_avg and center_y >= y_avg:
                            bbox["pos"] = "rd"
                        elif center_x < x_avg and center_y < y_avg:
                            bbox["pos"] = "ru"
                        else:
                            bbox["pos"] = "lu"



        # 侧面
        elif type == "side":
            if len(new_bbox_array) > 1:
                y_sum = 0
                for bbox in new_bbox_array:
                    y_sum += bbox["bbox"][1] + bbox["bbox"][3] / 2
                y_avg = y_sum / len(new_bbox_array)

                # 贴标签
                for bbox in new_bbox_array:
                    center_y = bbox["bbox"][1] + bbox["bbox"][3] / 2
                    if center_y >= y_avg:
                        bbox["pos"] = "d"
                    else:
                        bbox["pos"] = "u"
            elif len(new_bbox_array) == 1:
                # 根据3
                standard_tooth_array = tooth_dict["3"]
                # 侧面是2颗
                if len(standard_tooth_array) == 2:
                    y_sum = 0
                    for bbox in standard_tooth_array:
                        y_sum += bbox["bbox"][1] + bbox["bbox"][3] / 2
                    y_avg = y_sum / len(standard_tooth_array)

                    for bbox in new_bbox_array:
                        center_y = bbox["bbox"][1] + bbox["bbox"][3] / 2
                        if center_y >= y_avg:
                            bbox["pos"] = "d"
                        else:
                            bbox["pos"] = "u"


        # 口内 根据x,如果是下颌, 大左， 小右
        # 口内 根据x,如果是上颌, 大右，下左(图片会反转吗)
        elif type == "inside":
            if len(new_bbox_array) > 1:
                # 计算坐标中心
                x_sum = 0
                for bbox in new_bbox_array:
                    x_sum += bbox["bbox"][0] + bbox["bbox"][2] / 2
                x_avg = x_sum / len(new_bbox_array)

                # 贴标签
                for bbox in new_bbox_array:
                    center_x = bbox["bbox"][0] + bbox["bbox"][2] / 2
                    if center_x >= x_avg:
                        bbox["pos"] = "l"
                    else:
                        bbox["pos"] = "r"
            elif len(new_bbox_array) == 1:
                # 根据1
                standard_tooth_array = tooth_dict["1"]
                # 侧面是2颗
                if len(standard_tooth_array) == 2:
                    x_sum = 0
                    for bbox in standard_tooth_array:
                        x_sum += bbox["bbox"][0] + bbox["bbox"][2] / 2
                    x_avg = x_sum / len(standard_tooth_array)

                    for bbox in new_bbox_array:
                        center_x = bbox["bbox"][0] + bbox["bbox"][2] / 2
                        if center_x >= x_avg:
                            bbox["pos"] = "l"
                        else:
                            bbox["pos"] = "r"

    final_result = []
    for new_bbox_array in tooth_dict.values():
        final_result.extend(new_bbox_array)
    # print(final_result)
    return final_result


def visualization(img_path, result):
    """
    显示结果
    :param img_path:
    :return:
    """
    # 读取图像
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(draw_bbox_mask(img, filter_pip_line(result, "front")))
    plt.show()


if __name__ == '__main__':
    # front_and_side_predictor = pdx.deploy.Predictor('front_and_side_inference_model/inference_model')
    inside_predictor = pdx.deploy.Predictor('inside_inference_model/inference_model')
    inputpath = "./inside_image_set"
    i = 0
    listdir = os.listdir(inputpath)
    for filename in listdir:
        file = os.path.join(inputpath, filename)
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = inside_predictor.predict(img)
        plt.figure()
        result_new = filter_pip_line(result, "inside")
        print(result_new)
        plt.imshow(draw_bbox_mask(img, result_new))
        plt.savefig('./result_img/{}_processed.png'.format(filename), dpi=300)
