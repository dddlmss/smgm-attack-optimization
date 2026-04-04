# yolo_attack.py
import os
import colorsys
import cv2
import numpy as np
import tensorflow as tf

from timeit import default_timer as timer

from keras import backend as K
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

from metaheuristics import MetaheuristicAttacks
from results_logger import append_result_row, build_result_row

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def letterbox_image_1(image, w, h, nw, nh):
    iw, ih = image.size
    cbox = [(w - nw) // 2, (h - nh) // 2, (w - nw) // 2 + nw, (h - nh) // 2 + nh]
    return image.crop(cbox)

class Yolo4(MetaheuristicAttacks):
    def __init__(self, score, iou, anchors_path, classes_path, model_path, gpu_num=1):
        self.score = score
        self.iou = iou
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.model_path = model_path
        self.gpu_num = gpu_num
        self.load_yolo()

    def get_class(self):
        with open(os.path.expanduser(self.classes_path)) as f:
            return [c.strip() for c in f.readlines()]

    def get_anchors(self):
        with open(os.path.expanduser(self.anchors_path)) as f:
            anchors = [float(x) for x in f.readline().split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_yolo(self):
        self.class_names = self.get_class()
        self.anchors = self.get_anchors()

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), self.colors))

        self.sess = K.get_session()
        self.yolo4_model = yolo4_body(Input(shape=(608, 608, 3)), num_anchors // 3, num_classes)
        self.yolo4_model.load_weights(os.path.expanduser(self.model_path))

        self.input_image_shape = K.placeholder(shape=(2,))
        self.boxes, self.scores, self.classes = yolo_eval(
            self.yolo4_model.output,
            self.anchors,
            len(self.class_names),
            self.input_image_shape,
            score_threshold=self.score
        )

        print(f"{self.model_path} model, anchors, and classes loaded.")

    def close_session(self):
        self.sess.close()

    def _select_topk_boxes_by_score(self, out_boxes, out_scores, k=3):
        if len(out_boxes) == 0:
            return []
        k = min(k, len(out_boxes))
        top_idx = np.argsort(out_scores)[::-1][:k]
        return [out_boxes[i] for i in top_idx]

    def _apply_mask_from_individual(self, original_image, individual, selected_boxes, grid_size=4):
        adv = np.copy(original_image)

        if len(selected_boxes) == 0:
            return adv

        coeffs_per_box = grid_size * grid_size

        for box_idx, box in enumerate(selected_boxes):
            top, left, bottom, right = box

            top = max(0, int(np.floor(top)))
            left = max(0, int(np.floor(left)))
            bottom = min(adv.shape[1], int(np.ceil(bottom)))
            right = min(adv.shape[2], int(np.ceil(right)))

            if bottom <= top or right <= left:
                continue

            start = box_idx * coeffs_per_box
            end = start + coeffs_per_box
            cell_coeffs = individual[start:end].reshape(grid_size, grid_size).astype(np.float32)

            mask_resized = cv2.resize(
                cell_coeffs,
                (right - left, bottom - top),
                interpolation=cv2.INTER_LINEAR
            )

            for ch in range(3):
                adv[0, top:bottom, left:right, ch] += mask_resized

        return np.clip(adv, 0.0, 1.0)

    def _get_detection_summary(self, image_data, image):
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo4_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            }
        )
        return {
            "num_boxes": int(len(out_boxes)),
            "score_sum": float(np.sum(out_scores)) if len(out_scores) > 0 else 0.0,
            "boxes": out_boxes,
            "scores": out_scores,
            "classes": out_classes
        }

    def _save_adv_array_as_image(self, adv_array, w, h, nw, nh, iw, ih, save_path):
        img = adv_array[0] * 255.
        im = Image.fromarray(img.astype(np.uint8))
        im = letterbox_image_1(im, w, h, nw, nh)
        im = im.resize((iw, ih), Image.BICUBIC)
        im.save(save_path)

    def attack_with_metaheuristic(
        self,
        image,
        attack_name,
        attack_type,
        jpgfile,
        attack_output_dir,
        csv_path,
        top_k=2,
        grid_size=4,
        eps=0.08,
        ga_params=None,
        pso_params=None,
        de_params=None,
        model_image_size=(608, 608)
    ):
        boxed_image, w, h, nw, nh, iw, ih = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32') / 255.
        image_data = np.expand_dims(image_data, 0)
        original_image = np.copy(image_data)

        cost_function = tf.reduce_sum(self.scores)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo4_model.input: original_image,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            }
        )

        selected_boxes = self._select_topk_boxes_by_score(out_boxes, out_scores, k=top_k)

        if attack_name == "GA":
            adv_image, best_cost, runtime_sec = self.run_ga_attack(
                original_image=original_image,
                image=image,
                cost_function=cost_function,
                selected_boxes=selected_boxes,
                **ga_params
            )
        elif attack_name == "PSO":
            adv_image, best_cost, runtime_sec = self.run_pso_attack(
                original_image=original_image,
                image=image,
                cost_function=cost_function,
                selected_boxes=selected_boxes,
                **pso_params
            )
        elif attack_name == "DE":
            adv_image, best_cost, runtime_sec = self.run_de_attack(
                original_image=original_image,
                image=image,
                cost_function=cost_function,
                selected_boxes=selected_boxes,
                **de_params
            )
        else:
            raise ValueError("Unsupported attack_name. Use GA, PSO, or DE.")

        os.makedirs(attack_output_dir, exist_ok=True)
        save_path = os.path.join(attack_output_dir, os.path.basename(jpgfile))
        self._save_adv_array_as_image(adv_image, w, h, nw, nh, iw, ih, save_path)

        clean_summary = self._get_detection_summary(original_image, image)
        adv_summary = self._get_detection_summary(adv_image, image)

        row = build_result_row(
            image_name=os.path.basename(jpgfile),
            method=attack_name,
            attack_type=attack_type,
            clean_num_boxes=clean_summary["num_boxes"],
            adv_num_boxes=adv_summary["num_boxes"],
            clean_score_sum=clean_summary["score_sum"],
            adv_score_sum=adv_summary["score_sum"],
            runtime_sec=runtime_sec,
            original_image=original_image,
            adv_image=adv_image,
            top_k_boxes=top_k,
            grid_size=grid_size
        )

        append_result_row(csv_path, row)
        print(f"[{attack_name}] saved attacked image to: {save_path}")
        print(f"[{attack_name}] logged results to: {csv_path}")

    def detect_image(self, image, model_image_size=(608, 608)):
        boxed_image, w, h, nw, nh, iw, ih = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32') / 255.
        image_data = np.expand_dims(image_data, 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo4_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            }
        )

        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32')
        )
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = f'{predicted_class} {score:.2f}'
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for j in range(thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        return image
    
    def attack_with_gradient(
        self,
        image,
        attack_name,
        attack_type,
        jpgfile,
        attack_output_dir,
        csv_path,
        model_image_size=(608, 608),
        alpha=0.02,
        max_iter=20,
        stop_threshold=0.002
    ):
        boxed_image, w, h, nw, nh, iw, ih = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32') / 255.
        image_data = np.expand_dims(image_data, 0)
        original_image = np.copy(image_data)

        cost_function = tf.reduce_sum(self.scores)
        gradient_function = K.gradients(cost_function, self.yolo4_model.input)[0]

        pre_g = np.zeros_like(image_data)
        runtime_start = timer()

        for index in range(max_iter):
            cost, gradients = self.sess.run(
                [cost_function, gradient_function],
                feed_dict={
                    self.yolo4_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                }
            )

            if attack_name == "SMGM":
                pre_n = np.sign(pre_g)
                g = gradients
                n = np.sign(g)
                pre_g = g
                image_data -= alpha * (pre_n + n)
                image_data = np.clip(image_data, 0.0, 1.0)

            elif attack_name == "I-FGSM":
                n = np.sign(gradients)
                image_data -= alpha * n
                image_data = np.clip(image_data, 0.0, 1.0)

            else:
                raise ValueError("Unsupported gradient attack")

            print(f"[{attack_name}] iter:{index} cost:{float(cost):.6f}")

            if float(cost) < stop_threshold:
                break

        runtime_sec = timer() - runtime_start

        os.makedirs(attack_output_dir, exist_ok=True)
        save_path = os.path.join(attack_output_dir, os.path.basename(jpgfile))
        self._save_adv_array_as_image(image_data, w, h, nw, nh, iw, ih, save_path)

        clean_summary = self._get_detection_summary(original_image, image)
        adv_summary = self._get_detection_summary(image_data, image)

        row = build_result_row(
            image_name=os.path.basename(jpgfile),
            method=attack_name,
            attack_type=attack_type,
            clean_num_boxes=clean_summary["num_boxes"],
            adv_num_boxes=adv_summary["num_boxes"],
            clean_score_sum=clean_summary["score_sum"],
            adv_score_sum=adv_summary["score_sum"],
            runtime_sec=runtime_sec,
            original_image=original_image,
            adv_image=image_data
        )

        append_result_row(csv_path, row)
        print(f"[{attack_name}] saved attacked image to: {save_path}")
        print(f"[{attack_name}] logged results to: {csv_path}")
        
