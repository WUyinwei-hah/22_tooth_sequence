
import requests
import cv2
import io
import base64
import numpy as np
from PIL import Image
import paddlex as pdx
import matplotlib.pyplot as plt
import pylab
import matplotlib
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


def draw_bbox(image, bboxes, bbox_labels, color_map=None):
    import numpy as np
    _SMALL_OBJECT_AREA_THRESH = 1000
    height, width = image.shape[:2]
    default_font_scale = max(np.sqrt(height * width) // 900, .5)
    linewidth = max(default_font_scale / 40, 2)

    labels = list()
    if color_map is None:
        color_map = get_color_map_list(len(train_dataset.labels) + 2)[2:]
    else:
        color_map = np.asarray(color_map)
        if color_map.shape[0] != len(bbox_labels) or color_map.shape[1] != 3:
            raise Exception(
                "The shape for color_map is required to be {}x3, but recieved shape is {}x{}.".
                    format(len(bbox_labels), color_map.shape))
        if np.max(color_map) > 255 or np.min(color_map) < 0:
            raise ValueError(
                " The values in color_map should be within 0-255 range.")
    print(color_map)
    for index in range(len(bboxes)):
        # cname, bbox, score = dt['category'], dt['bbox'], dt['score']
        bbox = list(map(int, bboxes[index]))
        # bbox = bboxes[index]
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        xmax = xmin + w
        ymax = ymin + h
        # print(w,h)
        # print(bbox_labels[index])
        color = tuple(map(int, color_map[bbox_labels[index][0]]))

        # draw bbox
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color,
                              linewidth)

        # draw label
        text_pos = (xmin, ymin)
        # print(xmin,ymin)
        instance_area = w * h
        if (instance_area < _SMALL_OBJECT_AREA_THRESH or h < 40):
            if ymin >= height - 5:
                text_pos = (xmin, ymin)
            else:
                text_pos = (xmin, ymax)
        height_ratio = h / np.sqrt(height * width)
        font_scale = (np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2,
                              2) * 0.5 * default_font_scale)
        text = "{} {:.2f}".format(train_dataset.labels[bbox_labels[index][0]], 1)
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


def draw_bbox_mask(image, results, threshold=0.5, color_map=None):
    import numpy as np
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
        text = "{} {:.2f}".format(cname, score)
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
def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


if __name__ == '__main__':
    # 获取图片的base64编码格式
    img = requests.get(url='https://7778-wxxy-6go88mip80d3619e-1309739381.tcb.qcloud.la/userImg/10009.JPG?sign=ddfef6214b91f7dbb0ca7596162148f7')
    # response = requests.get('https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png')
    bytes_im = io.BytesIO(img.content)
    cv_im = cv2.cvtColor(np.array(Image.open(bytes_im)), cv2.COLOR_RGB2BGR)
    # img_file(List[np.ndarray or str], str or np.ndarray): 图像路径；或者是解码后的排列格式为（H, W, C）且类型为float32且为BGR格式的数组
    predictor = pdx.deploy.Predictor('front_and_side_inference_model')
    result = predictor.predict(cv_im)
    print(result)
    img_pre = draw_bbox_mask(cv_im, result, threshold=0.5)
    img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
    image = cv2.imencode('.jpg', img_pre)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    print(image_code)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img_pre)
    # pylab.show()

    # # 打印预测结果
    # print(r.json()["results"])