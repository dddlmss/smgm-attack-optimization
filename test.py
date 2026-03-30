import os
import colorsys
import random
import cv2

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import matplotlib.pyplot as plt
graph = tf.get_default_graph()

def letterbox_image_1(image, w, h, nw, nh):
    """
    resize image with unchanged aspect ratio using padding
    图像截取
    """
    iw, ih = image.size

    # if iw > ih:
    #     cbox = [(w-nw)//2, (h-nh)//2, w, (h-nh)//2 + nh]
    # else:
    cbox = [(w-nw)//2, (h-nh)//2, (w-nw)//2 + nw, (h-nh)//2 + nh]

    image_cropped = image.crop(cbox)

    return image_cropped
    
def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    # TODO: move this function to utils
    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    assert isinstance(p, float) or isinstance(p, int)
    norm = np.linalg.norm(x)
    norm = np.maximum(norm, 1 * small_constant)
    return (1. / norm) * x

class Yolo4(object):
    def get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_yolo(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.class_names = self.get_class()
        self.anchors = self.get_anchors()

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        self.sess = K.get_session()

        # Load model, or construct model and load weights.
        self.yolo4_model = yolo4_body(Input(shape=(608, 608, 3)), num_anchors // 3, num_classes)
        self.yolo4_model.load_weights(model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        if self.gpu_num >= 2:
            self.yolo4_model = multi_gpu_model(self.yolo4_model, gpus=self.gpu_num)

        self.input_image_shape = K.placeholder(shape=(2,))
        self.boxes, self.scores, self.classes = yolo_eval(self.yolo4_model.output, self.anchors,
                                                          len(self.class_names), self.input_image_shape,
                                                          score_threshold=self.score)

    def __init__(self, score, iou, anchors_path, classes_path, model_path, gpu_num=1):
        self.score = score
        self.iou = iou
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.model_path = model_path
        self.gpu_num = gpu_num
        self.load_yolo()

    def close_session(self):
        self.sess.close()

    def detect_image(self, image, model_image_size=(608, 608)):
        start = timer()

        boxed_image, w, h, nw, nh, iw, ih = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo4_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print(out_boxes, out_scores, out_classes)

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            print(out_classes)
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            # all_score.append(score)
            
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            
            top, left, bottom, right = box
            if 0 <= c <= 5:
                with open('train.txt', 'a', encoding='utf-8') as f:
                    f.write(str("{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}".format(left, top, right, bottom, c)) + ' ')
            if c==7:
                c_1 = c - 1
                with open('train.txt', 'a', encoding='utf-8') as f:
                    f.write(str("{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}".format(left, top, right, bottom, c_1)) + ' ')
                
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        # print(end - start)
        return image

    def Attack(self, image, attack_name, count, jpgfile, model_image_size=(608, 608)):
        # sess = K.get_session()
        global graph
        ori_image = image # 扰动截取之用
        pixdata_1 = ori_image.load()
        
        boxed_image, w, h, nw, nh, iw, ih = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        original_image = np.copy(image_data)
        # with graph.as_default():
        object_hack = 2  # 该参数为要攻击的目标类别
        A = self.classes >= object_hack
        B = self.classes <= object_hack
        hack_scores = tf.boolean_mask(self.scores, A & B)
        # a, b = self.sess.run(
        # [A, hack_scores],
        #     feed_dict={
        #         self.yolo4_model.input: image_data,
        #         self.input_image_shape: [image.size[1], image.size[0]],
        #         K.learning_phase(): 0
        #     })
        # cost_function = tf.add(hack_scores[0], hack_scores[1])  # 报错
        # cost_function = tf.reduce_sum(hack_scores)  # 跑通代码(指定目标攻击为隐身)
        cost_function = tf.reduce_sum(self.scores)  # 跑通代码(全部目标攻击为隐身)
        print("cost_function:{}".format(cost_function))

        # 수정 !!!
        # 이거 아래로 옮겼어요 gradient_function = K.gradients(cost_function, self.yolo4_model.input)[0]
        
        cost = 1
        alpha = 0.02
        n = 0
        ne = 0
        index = 0
        top_min, left_min, bottom_max, right_max = 0, 0, 0, 0 # 初始化坐标
        top_list= []
        left_list= []
        bottom_list= []
        right_list= []
        # 最大改变幅度
        max_change_above = original_image + 0.1
        max_change_below = original_image - 0.1
        # 初始化梯度
        pre_g = np.zeros(image_data.shape)
        D = np.zeros(image_data.shape)
        data_adv = np.zeros(image_data.shape)
        gradients_m = np.zeros(image_data.shape)

        
        # -------------------------
        # Gradient-based methods
        # -------------------------
        if attack_name in ['I-FGSM', 'Jung', 'MI-FGSM', 'CI-FGSM', 'AO2AM', 'AI-FGSM', 'SMGM']:
            gradient_function = K.gradients(cost_function, self.yolo4_model.input)[0]

            # define missing variables used below
            x0 = np.copy(original_image)
            e = alpha



            # 主要攻击循环
            while cost > 0.002:
            # for i in range(0, 10):
                img = image_data[0]
                img *= 255. 
                im = Image.fromarray(img.astype(np.uint8))
                im = letterbox_image_1(im, w, h, nw, nh)
                im = im.resize((iw, ih), Image.BICUBIC) # 填充图像
                # '''裁剪扰动'''
                # if attack_name == 'SMGM-APS':
                    # im = im.convert("RGB")
                    # pixdata = im.load()
                    # for i_width in range(iw):#遍历图片的所有像素
                        # for j_height in range(ih):
                            # if i_width < left_min or i_width > right_max or j_height < top_min or j_height > bottom_max:
                                # pixdata[i_width,j_height] = pixdata_1[i_width,j_height]


                # 수정 부분!!!! 

                attack_type = "untargeted"
                save_dir = os.path.join("output", f"{attack_name.lower()}_{attack_type}")
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, os.path.basename(jpgfile))
                im.save(save_path)

                # 再次打开图片
                im = Image.open(save_path)

                # 수정 부분 끝!!

                
                im, w, h, nw, nh, iw, ih = letterbox_image(im, tuple(reversed(model_image_size)))
                image_data = np.array(im, dtype='float32')
                image_data /= 255.
                image_data = np.expand_dims(image_data, 0)
                # 计算梯度
                # with graph.as_default():
                cost, gradients, out_classes, out_boxes = self.sess.run(
                    [cost_function, gradient_function, self.classes, self.boxes],
                    feed_dict={
                        self.yolo4_model.input: image_data,
                        self.input_image_shape: [image.size[1], image.size[0]],
                        K.learning_phase(): 0
                    })
                print("batch:{} Cost: {:.8}".format(index, cost))
                for i, c in reversed(list(enumerate(out_classes))):
                    box = out_boxes[i]
                    top, left, bottom, right = box
                    top_list.append(top)
                    left_list.append(left)
                    bottom_list.append(bottom)
                    right_list.append(right)
                if not top_list and index == 0:
                    top_min, left_min, bottom_max, right_max = 0, 0, 0, 0 # 初始化坐标
                else:
                    top_min = min(top_list)
                    left_min = min(left_list)
                    bottom_max = max(bottom_list)
                    right_max = max(right_list)

            
                # 计算噪声
                if attack_name == 'I-FGSM':
                    n = np.sign(gradients)
                    image_data -= n * e
                    image_data = np.clip(image_data, 0, 1.0)
                if attack_name == 'Jung':
                    eps = 8/255.0
                    image_data = np.clip(x0 + np.random.uniform(-eps, eps, size=x0.shape), 0.0, 1.0)
                    n = np.sign(gradients)
                    image_data -= e * n
                    # 投影到 L∞ 球 & 像素裁剪
                    image_data = np.minimum(x0 + eps, np.maximum(x0 - eps, image_data))
                    image_data = np.clip(image_data, 0.0, 1.0)
                if attack_name == 'MI-FGSM':
                    g = pre_g + gradients / np.linalg.norm(gradients, ord=1, axis=2)
                    pre_g = g
                    n = np.sign(g)
                    image_data -= n * e
                    image_data = np.clip(image_data, 0, 1.0)
                if attack_name == 'CI-FGSM':
                    gradients = gradients - 0.5 * gradients_m
                    gradients_m = gradients_m + 0.9 *(gradients- 0.9 * gradients_m)
                    n = np.linalg.norm(gradients_m, ord=1, axis=2)
                    n = np.clip(n, max_change_below, max_change_above)
                    image_data -= n * e
                    image_data = np.clip(image_data, 0, 1.0)
                if attack_name == 'AO2AM':
                    m = pow(gradients, 2)
                    n = gradients / ((m + 0.00000001) ** 0.5)
                    image_data -= n * e
                    image_data = np.clip(image_data, 0, 1.0)
                if attack_name == 'AI-FGSM':
                    g = 0.1 * pre_g + 0.9 * pow(np.sign(gradients), 2)
                    pre_g = g
                    n = gradients / ((g + 0.00000001) ** 0.5)
                    image_data -= n * e
                    image_data = np.clip(image_data, 0, 1.0)
                if attack_name == 'SMGM':
                    pre_n = np.sign(pre_g)
                    g = gradients
                    n = np.sign(g)
                    pre_g = g
                    image_data -= (pre_n * alpha + n * alpha)
                    image_data = np.clip(image_data, 0, 1.0)
                index += 1
                if cost < 0.002:
                    break
            
            return 0
    
       # -------------------------
        # PSO
        # -------------------------
        elif attack_name == 'PSO':
            image_data = self._run_pso_attack(
                original_image=original_image,
                image=image,
                cost_function=cost_function
            )
            return 0

        # -------------------------
        # GA
        # -------------------------
        elif attack_name == 'GA':
            image_data = self._run_ga_attack(
                original_image=original_image,
                image=image,
                cost_function=cost_function
            )
            return 0

        else:
            raise ValueError("Unsupported attack_name")
        

    def TargetAttack(self, image, attack_name, count, jpgfile, model_image_size=(608, 608)):
        global graph
        ori_image = image # 扰动截取之用
        pixdata_1 = ori_image.load()
        
        boxed_image, w, h, nw, nh, iw, ih = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        # 增加额外维度
        image_data = np.expand_dims(image_data, 0)
        original_image = np.copy(image_data)
        out_scores, out_classes = self.sess.run(
            [self.scores, self.classes],
            feed_dict={
                self.yolo4_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        # 定义maxcost
        maxcost = 0
        object_hack = 2  # 该参数为要攻击的目标类别
        object_target = 5  # 该参数为攻击成的目标类别
        A = self.classes >= object_target
        B = self.classes <= object_target
        hack_scores = tf.boolean_mask(self.scores, A & B)
        cost_function = tf.reduce_sum(hack_scores)  # 跑通代码，定向攻击
        gradient_function = K.gradients(cost_function, self.yolo4_model.input)[0]
        cost = 1
        alpha = 0.02
        n = 0
        r = 0
        index = 0
        top_min, left_min, bottom_max, right_max = 0, 0, 0, 0 # 初始化坐标
        top_list= []
        left_list= []
        bottom_list= []
        right_list= []
        # 最大改变幅度
        max_change_above = original_image + 0.1
        max_change_below = original_image - 0.1
        # 初始设置
        pre_g = np.zeros(image_data.shape)
        gradients_m = np.zeros(image_data.shape)

        # 主要攻击循环
        if object_hack in list(out_classes):
            for i in range(0, 20):
                img = image_data[0]
                img *= 255.
                im = Image.fromarray(img.astype(np.uint8))
                im = letterbox_image_1(im, w, h, nw, nh)
                im = im.resize((iw, ih), Image.BICUBIC)
                # '''裁剪扰动'''
                # if attack_name == 'SMGM-APS':
                    # im = im.convert("RGB")
                    # pixdata = im.load()
                    # for i_width in range(iw):#遍历图片的所有像素
                        # for j_height in range(ih):
                            # if i_width < left_min or i_width > right_max or j_height < top_min or j_height > bottom_max:
                                # pixdata[i_width,j_height] = pixdata_1[i_width,j_height]



            
                # 수정 부분!!!
            
                attack_type = "targeted"
                save_dir = os.path.join("output", f"{attack_name.lower()}_{attack_type}")
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, os.path.basename(jpgfile))
                im.save(save_path)


       
                im = Image.open(save_path)

                # 수정 부분!!!



                im, w, h, nw, nh, iw, ih = letterbox_image(im, tuple(reversed(model_image_size)))
                image_data = np.array(im, dtype='float32')
                image_data /= 255.
                image_data = np.expand_dims(image_data, 0)
                # 计算梯度
                cost, gradients, out_scores, out_classes, out_boxes = self.sess.run(
                    [cost_function, gradient_function, self.scores, self.classes, self.boxes],
                    feed_dict={
                        self.yolo4_model.input: image_data,
                        self.input_image_shape: [image.size[1], image.size[0]],
                        K.learning_phase(): 0})
                for i, c in reversed(list(enumerate(out_classes))):
                    score = out_scores[i]
                    if c == 2:
                        maxcost += score
                    box = out_boxes[i]
                    top, left, bottom, right = box
                    top_list.append(top)
                    left_list.append(left)
                    bottom_list.append(bottom)
                    right_list.append(right)
                print(maxcost)
                if not top_list and index == 0: # 原始图像没有目标且列表为空
                    top_min, left_min, bottom_max, right_max = 0, 0, 0, 0 # 初始化坐标
                else:
                    top_min = min(top_list)
                    left_min = min(left_list)
                    bottom_max = max(bottom_list)
                    right_max = max(right_list)
                # 计算噪声
                if attack_name == 'I-FGSM':
                    n = np.sign(gradients)
                    image_data -= n * e
                    image_data = np.clip(image_data, 0, 1.0)
                if attack_name == 'Jung':
                    eps = 8/255.0
                    image_data = np.clip(x0 + np.random.uniform(-eps, eps, size=x0.shape), 0.0, 1.0)
                    n = np.sign(gradients)
                    image_data -= e * n
                    # 投影到 L∞ 球 & 像素裁剪
                    image_data = np.minimum(x0 + eps, np.maximum(x0 - eps, image_data))
                    image_data = np.clip(image_data, 0.0, 1.0)
                if attack_name == 'MI-FGSM':
                    g = pre_g + gradients / np.linalg.norm(gradients, ord=1, axis=2)
                    pre_g = g
                    n = np.sign(g)
                    image_data -= n * e
                    image_data = np.clip(image_data, 0, 1.0)
                if attack_name == 'CI-FGSM':
                    gradients = gradients - 0.5 * gradients_m
                    gradients_m = gradients_m + 0.9 *(gradients- 0.9 * gradients_m)
                    n = np.linalg.norm(gradients_m, ord=1, axis=2)
                    n = np.clip(n, max_change_below, max_change_above)
                    image_data -= n * e
                    image_data = np.clip(image_data, 0, 1.0)
                if attack_name == 'AO2AM':
                    m = pow(gradients, 2)
                    n = gradients / ((m + 0.00000001) ** 0.5)
                    image_data -= n * e
                    image_data = np.clip(image_data, 0, 1.0)
                if attack_name == 'AI-FGSM':
                    g = 0.1 * pre_g + 0.9 * pow(np.sign(gradients), 2)
                    pre_g = g
                    n = gradients / ((g + 0.00000001) ** 0.5)
                    image_data -= n * e
                    image_data = np.clip(image_data, 0, 1.0)
                if attack_name == 'SMGM':
                    pre_n = np.sign(pre_g)
                    g = gradients
                    n = np.sign(g)
                    pre_g = g
                    image_data -= (pre_n * alpha + n * alpha)
                    image_data = np.clip(image_data, 0, 1.0)

                print("batch:{} Cost: {:.8}".format(index, cost))
                # with open('logs/AM2-FGSM_targeted.txt', 'a', encoding='utf-8') as f:
                    # f.write(str("{:.8}".format(cost)) + '\n')
                index += 1
                if maxcost < 0.5:
                    break
                maxcost = 0
        return 0



# 수정 부분!!!

if __name__ == '__main__':
    import glob
    import os
    from PIL import Image

    model_path = 'yolo4_weight.h5'
    anchors_path = 'model_data/yolo4_anchors.txt'
    classes_path = 'model_data/coco_classes.txt'
    score = 0.5
    iou = 0.5
    model_image_size = (608, 608)

    # -----------------------------
    # Easy switches (VERY CLEAR NOW)
    # -----------------------------
    MODE = "attack"               # "attack" or "detect"
    attack_name = "SMGM"          # later: "PSO"
    ATTACK_TYPE = "untargeted"    # "untargeted" or "targeted"

    # original images
    input_path = r"test\original\*.jpg"

    # folders (automatically match attack type)
    attack_output_dir = os.path.join("output", f"{attack_name.lower()}_{ATTACK_TYPE}")
    detect_output_dir = os.path.join("test", "result", f"{attack_name.lower()}_{ATTACK_TYPE}")

    os.makedirs(attack_output_dir, exist_ok=True)
    os.makedirs(detect_output_dir, exist_ok=True)

    yolo4_model = Yolo4(score, iou, anchors_path, classes_path, model_path)

    if MODE == "attack":
        print(f"[INFO] Attack type: {ATTACK_TYPE}")
        print(f"[INFO] Saving attacked images to: {attack_output_dir}")

        count = 0
        for jpgfile in glob.glob(input_path):
            print(f"[ATTACK] {jpgfile}")
            img = Image.open(jpgfile)

            if ATTACK_TYPE == "targeted":
                yolo4_model.TargetAttack(
                    img,
                    attack_name,
                    count,
                    jpgfile,
                    model_image_size=model_image_size
                )
            elif ATTACK_TYPE == "untargeted":
                yolo4_model.Attack(
                    img,
                    attack_name,
                    count,
                    jpgfile,
                    model_image_size=model_image_size
                )
            else:
                raise ValueError("ATTACK_TYPE must be 'targeted' or 'untargeted'")

            count += 1
            print(f"[DONE] {count} images processed.")

    elif MODE == "detect":
        detect_input_path = os.path.join("output", f"{attack_name.lower()}_{ATTACK_TYPE}", "*.jpg")

        print(f"[INFO] Attack type: {ATTACK_TYPE}")
        print(f"[INFO] Reading attacked images from: {detect_input_path}")
        print(f"[INFO] Saving detected images to: {detect_output_dir}")

        for jpgfile in glob.glob(detect_input_path):
            print(f"[DETECT] {jpgfile}")
            img = Image.open(jpgfile)
            img = yolo4_model.detect_image(img)
            img.save(os.path.join(detect_output_dir, os.path.basename(jpgfile)))

    else:
        print("MODE must be 'attack' or 'detect'")

    yolo4_model.close_session()