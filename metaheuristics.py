import numpy as np
from timeit import default_timer as timer
from keras import backend as K


class MetaheuristicAttacks:
    def _random_individual(self, num_boxes, grid_size=4, eps=0.08):
        num_vars = num_boxes * grid_size * grid_size
        return np.random.uniform(-eps, eps, size=(num_vars,)).astype(np.float32)

    # =========================================================
    # Global overlap patch representation
    # =========================================================
    def _build_overlap_patch_specs(self, selected_boxes, original_image, patch_grid=16):
        """
        Build a global patch grid over the full image.
        Keep only patches that overlap with at least one selected bbox.
        """
        H = original_image.shape[1]
        W = original_image.shape[2]

        if len(selected_boxes) == 0:
            return []

        patch_h = int(np.ceil(H / patch_grid))
        patch_w = int(np.ceil(W / patch_grid))

        active_patches = []

        for r in range(patch_grid):
            for c in range(patch_grid):
                top = r * patch_h
                bottom = min((r + 1) * patch_h, H)
                left = c * patch_w
                right = min((c + 1) * patch_w, W)

                overlaps = False
                for box in selected_boxes:
                    b_top, b_left, b_bottom, b_right = box

                    b_top = max(0, int(np.floor(b_top)))
                    b_left = max(0, int(np.floor(b_left)))
                    b_bottom = min(H, int(np.ceil(b_bottom)))
                    b_right = min(W, int(np.ceil(b_right)))

                    inter_top = max(top, b_top)
                    inter_left = max(left, b_left)
                    inter_bottom = min(bottom, b_bottom)
                    inter_right = min(right, b_right)

                    if inter_bottom > inter_top and inter_right > inter_left:
                        overlaps = True
                        break

                if overlaps:
                    active_patches.append({
                        "row": r,
                        "col": c,
                        "top": top,
                        "bottom": bottom,
                        "left": left,
                        "right": right
                    })

        return active_patches

    def _patch_key(self, patch):
        return (patch["row"], patch["col"])

    def _get_boundary_patch_indices(self, active_patches):
        active_set = {self._patch_key(p) for p in active_patches}
        boundary = []

        for i, p in enumerate(active_patches):
            r, c = p["row"], p["col"]
            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            if any(n not in active_set for n in neighbors):
                boundary.append(i)

        return boundary

    def _choose_adaptive_patch_grid(self, selected_boxes, original_image, min_grid=10, max_grid=24):
        """
        Choose patch grid resolution based on average selected bbox size.
        Smaller boxes -> finer grid
        Larger boxes -> coarser grid
        """
        if selected_boxes is None or len(selected_boxes) == 0:
            return 16

        H = original_image.shape[1]
        W = original_image.shape[2]
        img_area = float(H * W)

        areas = []
        for box in selected_boxes:
            top, left, bottom, right = box
            h = max(1.0, float(bottom) - float(top))
            w = max(1.0, float(right) - float(left))
            areas.append(h * w)

        mean_ratio = np.mean(areas) / max(img_area, 1.0)

        if mean_ratio < 0.01:
            return max_grid
        elif mean_ratio < 0.03:
            return min(max_grid, 20)
        elif mean_ratio < 0.08:
            return 16
        else:
            return min_grid

    def _apply_active_patch_values(self, original_image, patch_values, active_patches):
        """
        Supports:
            patch_values shape [num_active_patches]      -> scalar patch value
            patch_values shape [num_active_patches, 3]   -> RGB patch value
        """
        adv = np.copy(original_image)

        if len(active_patches) == 0:
            return adv

        patch_values = np.asarray(patch_values, dtype=np.float32)
        scalar_mode = (patch_values.ndim == 1)

        for i, patch in enumerate(active_patches):
            top = patch["top"]
            bottom = patch["bottom"]
            left = patch["left"]
            right = patch["right"]

            if scalar_mode:
                adv[0, top:bottom, left:right, :] += float(patch_values[i])
            else:
                adv[0, top:bottom, left:right, 0] += float(patch_values[i, 0])
                adv[0, top:bottom, left:right, 1] += float(patch_values[i, 1])
                adv[0, top:bottom, left:right, 2] += float(patch_values[i, 2])

        return np.clip(adv, 0.0, 1.0)

    # =========================================================
    # IoU utilities
    # =========================================================
    def _compute_iou(self, box_a, box_b):
        a_top, a_left, a_bottom, a_right = box_a
        b_top, b_left, b_bottom, b_right = box_b

        inter_top = max(float(a_top), float(b_top))
        inter_left = max(float(a_left), float(b_left))
        inter_bottom = min(float(a_bottom), float(b_bottom))
        inter_right = min(float(a_right), float(b_right))

        inter_h = max(0.0, inter_bottom - inter_top)
        inter_w = max(0.0, inter_right - inter_left)
        inter_area = inter_h * inter_w

        area_a = max(0.0, float(a_bottom) - float(a_top)) * max(0.0, float(a_right) - float(a_left))
        area_b = max(0.0, float(b_bottom) - float(b_top)) * max(0.0, float(b_right) - float(b_left))
        union = area_a + area_b - inter_area

        if union <= 1e-12:
            return 0.0
        return inter_area / union

    def _mean_best_iou(self, clean_boxes, adv_boxes):
        if clean_boxes is None or len(clean_boxes) == 0:
            return 0.0
        if adv_boxes is None or len(adv_boxes) == 0:
            return 0.0

        best_ious = []
        for clean_box in clean_boxes:
            best = 0.0
            for adv_box in adv_boxes:
                iou = self._compute_iou(clean_box, adv_box)
                if iou > best:
                    best = iou
            best_ious.append(best)

        return float(np.mean(best_ious)) if best_ious else 0.0

    # =========================================================
    # Shape dictionary over active patches
    # =========================================================
    def _normalize_mask(self, vec):
        vec = vec.astype(np.float32)
        max_abs = np.max(np.abs(vec))
        if max_abs < 1e-12:
            return None
        return vec / max_abs

    def _build_shape_dictionary(self, active_patches):
        """
        Build structured masks over the active overlap patches.

        Returns:
            shape_dict: ndarray [num_shapes, num_active_patches]
            shape_names: list[str]
        """
        n = len(active_patches)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32), []

        rows = np.array([p["row"] for p in active_patches], dtype=np.int32)
        cols = np.array([p["col"] for p in active_patches], dtype=np.int32)

        uniq_rows = sorted(np.unique(rows).tolist())
        uniq_cols = sorted(np.unique(cols).tolist())

        boundary_idx = set(self._get_boundary_patch_indices(active_patches))
        interior_idx = set(range(n)) - boundary_idx

        shape_list = []
        shape_names = []
        seen = set()

        def add_shape(name, vec):
            normed = self._normalize_mask(vec)
            if normed is None:
                return
            key = tuple(np.round(normed, 4).tolist())
            if key in seen:
                return
            seen.add(key)
            shape_list.append(normed)
            shape_names.append(name)

        add_shape("all_active", np.ones(n, dtype=np.float32))

        add_shape(
            "checkerboard",
            np.array([1.0 if ((p["row"] + p["col"]) % 2 == 0) else -1.0 for p in active_patches], dtype=np.float32)
        )

        boundary_vec = np.zeros(n, dtype=np.float32)
        for i in boundary_idx:
            boundary_vec[i] = 1.0
        add_shape("boundary", boundary_vec)

        interior_vec = np.zeros(n, dtype=np.float32)
        for i in interior_idx:
            interior_vec[i] = 1.0
        add_shape("interior", interior_vec)

        row_mid = float(np.median(rows))
        col_mid = float(np.median(cols))

        add_shape("top_half", np.array([1.0 if p["row"] <= row_mid else 0.0 for p in active_patches], dtype=np.float32))
        add_shape("bottom_half", np.array([1.0 if p["row"] >= row_mid else 0.0 for p in active_patches], dtype=np.float32))
        add_shape("left_half", np.array([1.0 if p["col"] <= col_mid else 0.0 for p in active_patches], dtype=np.float32))
        add_shape("right_half", np.array([1.0 if p["col"] >= col_mid else 0.0 for p in active_patches], dtype=np.float32))

        for r in uniq_rows:
            add_shape(
                f"row_{r}",
                np.array([1.0 if p["row"] == r else 0.0 for p in active_patches], dtype=np.float32)
            )

        for c in uniq_cols:
            add_shape(
                f"col_{c}",
                np.array([1.0 if p["col"] == c else 0.0 for p in active_patches], dtype=np.float32)
            )

        center_blob = np.zeros(n, dtype=np.float32)
        for i, p in enumerate(active_patches):
            if abs(p["row"] - row_mid) <= 1 and abs(p["col"] - col_mid) <= 1:
                center_blob[i] = 1.0
        add_shape("center_blob", center_blob)

        sorted_idx = sorted(range(n), key=lambda i: (active_patches[i]["row"], active_patches[i]["col"]))
        if len(sorted_idx) > 0:
            anchors = [
                sorted_idx[0],
                sorted_idx[len(sorted_idx) // 2],
                sorted_idx[-1]
            ]
            for k, anchor_idx in enumerate(anchors):
                ar = active_patches[anchor_idx]["row"]
                ac = active_patches[anchor_idx]["col"]
                block = np.zeros(n, dtype=np.float32)
                for i, p in enumerate(active_patches):
                    if abs(p["row"] - ar) <= 1 and abs(p["col"] - ac) <= 1:
                        block[i] = 1.0
                add_shape(f"local_block_{k}", block)

        if len(shape_list) == 0:
            return np.zeros((0, n), dtype=np.float32), []

        return np.stack(shape_list, axis=0).astype(np.float32), shape_names

    def _compose_patch_values_from_shapes(self, coeffs, shape_dict, eps=0.08, rgb_mode=True):
        """
        If rgb_mode=True:
            coeffs shape = [num_shapes * 3]
            output shape = [num_active_patches, 3]

        Else:
            coeffs shape = [num_shapes]
            output shape = [num_active_patches]
        """
        num_shapes = shape_dict.shape[0]

        if not rgb_mode:
            patch_values = np.dot(coeffs, shape_dict)
            patch_values = np.clip(patch_values, -eps, eps).astype(np.float32)
            return patch_values

        coeffs = np.asarray(coeffs, dtype=np.float32).reshape(num_shapes, 3)
        ch0 = np.dot(coeffs[:, 0], shape_dict)
        ch1 = np.dot(coeffs[:, 1], shape_dict)
        ch2 = np.dot(coeffs[:, 2], shape_dict)

        patch_values = np.stack([ch0, ch1, ch2], axis=1)
        patch_values = np.clip(patch_values, -eps, eps).astype(np.float32)
        return patch_values

    # =========================================================
    # Structured init in shape-coefficient space
    # =========================================================
    def _find_shape_indices(self, shape_names, prefix):
        return [i for i, name in enumerate(shape_names) if name.startswith(prefix)]

    def _structured_shape_initialization(
        self,
        shape_dict,
        shape_names,
        swarm_size=20,
        eps=0.08,
        rgb_mode=True
    ):
        """
        Structured initialization in shape-coefficient space.
        If rgb_mode=True, each shape has 3 channel coefficients.
        """
        num_shapes = shape_dict.shape[0]
        if num_shapes == 0:
            dim = 0
            return np.zeros((swarm_size, dim), dtype=np.float32)

        dim = num_shapes if not rgb_mode else num_shapes * 3
        particles = []

        boundary_idx = self._find_shape_indices(shape_names, "boundary")
        checker_idx = self._find_shape_indices(shape_names, "checkerboard")
        row_idx = self._find_shape_indices(shape_names, "row_")
        col_idx = self._find_shape_indices(shape_names, "col_")
        local_idx = self._find_shape_indices(shape_names, "local_block_")
        center_idx = self._find_shape_indices(shape_names, "center_blob")

        def empty_vec():
            return np.zeros(dim, dtype=np.float32)

        def set_shape(v, shape_idx, vals):
            if not rgb_mode:
                v[shape_idx] = vals[0]
            else:
                base = 3 * shape_idx
                v[base:base + 3] = np.array(vals, dtype=np.float32)

        for sign in [1.0, -1.0]:
            v = empty_vec()
            idx = np.random.randint(num_shapes)
            if rgb_mode:
                set_shape(v, idx, [sign * eps, 0.0, 0.0])
            else:
                set_shape(v, idx, [sign * eps])
            particles.append(v)

        if checker_idx:
            v = empty_vec()
            idx = checker_idx[0]
            if rgb_mode:
                set_shape(v, idx, [0.8 * eps, -0.6 * eps, 0.8 * eps])
            else:
                set_shape(v, idx, [0.8 * eps])
            particles.append(v)

        if boundary_idx:
            for sign in [1.0, -1.0]:
                v = empty_vec()
                idx = boundary_idx[0]
                if rgb_mode:
                    set_shape(v, idx, [sign * 0.8 * eps, sign * 0.4 * eps, -sign * 0.6 * eps])
                else:
                    set_shape(v, idx, [sign * 0.8 * eps])
                particles.append(v)

        if row_idx and col_idx:
            v = empty_vec()
            r_idx = np.random.choice(row_idx)
            c_idx = np.random.choice(col_idx)
            if rgb_mode:
                set_shape(v, r_idx, [0.7 * eps, 0.0, -0.4 * eps])
                set_shape(v, c_idx, [0.0, 0.7 * eps, -0.4 * eps])
            else:
                set_shape(v, r_idx, [0.7 * eps])
                set_shape(v, c_idx, [0.7 * eps])
            particles.append(v)

        if local_idx:
            for sign in [1.0, -1.0]:
                v = empty_vec()
                idx = np.random.choice(local_idx)
                if rgb_mode:
                    set_shape(v, idx, [sign * 0.8 * eps, -sign * 0.8 * eps, sign * 0.5 * eps])
                else:
                    set_shape(v, idx, [sign * 0.8 * eps])
                particles.append(v)

        if center_idx:
            v = empty_vec()
            idx = center_idx[0]
            if rgb_mode:
                set_shape(v, idx, [0.7 * eps, 0.7 * eps, -0.7 * eps])
            else:
                set_shape(v, idx, [0.7 * eps])
            particles.append(v)

        while len(particles) < swarm_size:
            mode = np.random.choice(["small_gauss", "wide_gauss", "uniform", "hybrid"])

            if mode == "small_gauss":
                v = np.random.normal(0.0, 0.01, size=dim).astype(np.float32)

            elif mode == "wide_gauss":
                v = np.random.normal(0.0, eps / 4.0, size=dim).astype(np.float32)

            elif mode == "uniform":
                v = np.random.uniform(-eps, eps, size=dim).astype(np.float32)

            else:
                v = np.random.normal(0.0, 0.01, size=dim).astype(np.float32)
                num_spikes = min(3, num_shapes)
                spike_shapes = np.random.choice(num_shapes, size=num_spikes, replace=False)

                for s in spike_shapes:
                    if not rgb_mode:
                        v[s] += np.random.choice([-1.0, 1.0]) * (0.6 * eps)
                    else:
                        base = 3 * s
                        v[base:base + 3] += np.random.choice([-1.0, 1.0], size=3) * (0.4 * eps)

            particles.append(np.clip(v, -eps, eps).astype(np.float32))

        return np.stack(particles[:swarm_size], axis=0)

    # =========================================================
    # Targeted candidate evaluation helpers
    # =========================================================
    def _extract_targeted_detection_terms(
        self,
        summary,
        clean_ref_boxes,
        topk=5,
        match_iou_thresh=0.10
    ):
        """
        Extract more attack-focused terms from detector output.
        """
        boxes = summary.get("boxes", []) if summary is not None else []
        scores = summary.get("scores", []) if summary is not None else []

        if scores is None:
            scores = []
        scores = np.asarray(scores, dtype=np.float32)

        if len(scores) == 0:
            topk_score_sum = 0.0
        else:
            topk_scores = np.sort(scores)[::-1][:topk]
            topk_score_sum = float(np.sum(topk_scores))

        matched_box_count = 0
        matched_score_sum = 0.0
        matched_best_ious = []

        if clean_ref_boxes is not None and len(clean_ref_boxes) > 0 and boxes is not None and len(boxes) > 0:
            for clean_box in clean_ref_boxes:
                best_iou = 0.0
                best_score = 0.0
                matched = False

                for adv_idx, adv_box in enumerate(boxes):
                    iou = self._compute_iou(clean_box, adv_box)
                    if iou > best_iou:
                        best_iou = iou
                        if adv_idx < len(scores):
                            best_score = float(scores[adv_idx])

                    if iou >= match_iou_thresh:
                        matched = True

                if matched:
                    matched_box_count += 1
                    matched_score_sum += best_score

                matched_best_ious.append(best_iou)

        mean_matched_iou = float(np.mean(matched_best_ious)) if len(matched_best_ious) > 0 else 0.0

        return {
            "topk_score_sum": topk_score_sum,
            "matched_box_count": float(matched_box_count),
            "matched_score_sum": float(matched_score_sum),
            "mean_matched_iou": mean_matched_iou,
        }

    # =========================================================
    # Candidate evaluation
    # =========================================================
    def _evaluate_candidate(
        self,
        original_image,
        individual,
        image,
        cost_function,
        selected_boxes=None,
        active_patches=None,
        shape_dict=None,
        clean_ref_boxes=None,
        grid_size=4,
        eps=0.08,
        rgb_mode=True,
        topk=5,
        match_iou_thresh=0.10,
        raw_cost_weight=0.10,
        topk_score_weight=1.20,
        matched_score_weight=1.50,
        matched_box_weight=0.90,
        matched_iou_weight=1.10,
        global_score_weight=0.20,
        global_box_weight=0.10,
        mag_weight=0.12
    ):
        if shape_dict is not None and active_patches is not None:
            patch_values = self._compose_patch_values_from_shapes(
                individual,
                shape_dict,
                eps=eps,
                rgb_mode=rgb_mode
            )
            adv = self._apply_active_patch_values(original_image, patch_values, active_patches)
            mag_penalty = float(np.mean(np.abs(patch_values)))

        elif active_patches is not None:
            adv = self._apply_active_patch_values(original_image, individual, active_patches)
            mag_penalty = float(np.mean(np.abs(individual)))

        else:
            adv = self._apply_mask_from_individual(
                original_image,
                individual,
                selected_boxes,
                grid_size=grid_size
            )
            mag_penalty = float(np.mean(np.abs(individual)))

        raw_cost = self.sess.run(
            cost_function,
            feed_dict={
                self.yolo4_model.input: adv,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            }
        )
        raw_cost = float(np.mean(raw_cost))

        summary = self._get_detection_summary(adv, image)
        num_boxes = float(summary.get("num_boxes", 0.0))
        score_sum = float(summary.get("score_sum", 0.0))

        targeted = self._extract_targeted_detection_terms(
            summary=summary,
            clean_ref_boxes=clean_ref_boxes,
            topk=topk,
            match_iou_thresh=match_iou_thresh
        )

        attack_loss = (
            raw_cost_weight * raw_cost
            + topk_score_weight * targeted["topk_score_sum"]
            + matched_score_weight * targeted["matched_score_sum"]
            + matched_box_weight * targeted["matched_box_count"]
            + matched_iou_weight * targeted["mean_matched_iou"]
            + global_score_weight * score_sum
            + global_box_weight * num_boxes
            + mag_weight * mag_penalty
        )

        fitness = -attack_loss
        return fitness, adv, raw_cost

    # =========================================================
    # NES over shape coefficients
    # =========================================================
    def _estimate_nes_gradient(
        self,
        theta,
        original_image,
        image,
        cost_function,
        active_patches,
        shape_dict,
        clean_ref_boxes,
        eps=0.08,
        rgb_mode=True,
        nes_sigma=0.10,
        nes_samples=6,
        topk=5,
        match_iou_thresh=0.10,
        raw_cost_weight=0.10,
        topk_score_weight=1.20,
        matched_score_weight=1.50,
        matched_box_weight=0.90,
        matched_iou_weight=1.10,
        global_score_weight=0.20,
        global_box_weight=0.10,
        mag_weight=0.12
    ):
        dim = len(theta)
        grad = np.zeros(dim, dtype=np.float32)

        if dim == 0:
            return grad

        for _ in range(nes_samples):
            noise = np.random.randn(dim).astype(np.float32)

            theta_pos = np.clip(theta + nes_sigma * noise, -eps, eps)
            theta_neg = np.clip(theta - nes_sigma * noise, -eps, eps)

            f_pos, _, _ = self._evaluate_candidate(
                original_image=original_image,
                individual=theta_pos,
                image=image,
                cost_function=cost_function,
                active_patches=active_patches,
                shape_dict=shape_dict,
                clean_ref_boxes=clean_ref_boxes,
                eps=eps,
                rgb_mode=rgb_mode,
                topk=topk,
                match_iou_thresh=match_iou_thresh,
                raw_cost_weight=raw_cost_weight,
                topk_score_weight=topk_score_weight,
                matched_score_weight=matched_score_weight,
                matched_box_weight=matched_box_weight,
                matched_iou_weight=matched_iou_weight,
                global_score_weight=global_score_weight,
                global_box_weight=global_box_weight,
                mag_weight=mag_weight
            )

            f_neg, _, _ = self._evaluate_candidate(
                original_image=original_image,
                individual=theta_neg,
                image=image,
                cost_function=cost_function,
                active_patches=active_patches,
                shape_dict=shape_dict,
                clean_ref_boxes=clean_ref_boxes,
                eps=eps,
                rgb_mode=rgb_mode,
                topk=topk,
                match_iou_thresh=match_iou_thresh,
                raw_cost_weight=raw_cost_weight,
                topk_score_weight=topk_score_weight,
                matched_score_weight=matched_score_weight,
                matched_box_weight=matched_box_weight,
                matched_iou_weight=matched_iou_weight,
                global_score_weight=global_score_weight,
                global_box_weight=global_box_weight,
                mag_weight=mag_weight
            )

            grad += ((f_pos - f_neg) / (2.0 * nes_sigma)) * noise

        grad /= float(nes_samples)
        return grad.astype(np.float32)

    def _local_refine_best_blackbox(
        self,
        best_theta,
        best_fitness,
        best_adv,
        best_cost,
        original_image,
        image,
        cost_function,
        active_patches,
        shape_dict,
        clean_ref_boxes,
        eps=0.08,
        rgb_mode=True,
        refine_steps=12,
        refine_sigma=0.05,
        refine_samples=10,
        refine_step=0.18,
        topk=5,
        match_iou_thresh=0.10,
        raw_cost_weight=0.10,
        topk_score_weight=1.20,
        matched_score_weight=1.50,
        matched_box_weight=0.90,
        matched_iou_weight=1.10,
        global_score_weight=0.20,
        global_box_weight=0.10,
        mag_weight=0.12
    ):
        theta = np.copy(best_theta)
        current_fitness = float(best_fitness)
        current_adv = np.copy(best_adv)
        current_cost = float(best_cost)

        for step_idx in range(refine_steps):
            grad = self._estimate_nes_gradient(
                theta=theta,
                original_image=original_image,
                image=image,
                cost_function=cost_function,
                active_patches=active_patches,
                shape_dict=shape_dict,
                clean_ref_boxes=clean_ref_boxes,
                eps=eps,
                rgb_mode=rgb_mode,
                nes_sigma=refine_sigma,
                nes_samples=refine_samples,
                topk=topk,
                match_iou_thresh=match_iou_thresh,
                raw_cost_weight=raw_cost_weight,
                topk_score_weight=topk_score_weight,
                matched_score_weight=matched_score_weight,
                matched_box_weight=matched_box_weight,
                matched_iou_weight=matched_iou_weight,
                global_score_weight=global_score_weight,
                global_box_weight=global_box_weight,
                mag_weight=mag_weight
            )

            proposal = np.clip(theta + refine_step * grad, -eps, eps)

            prop_fitness, prop_adv, prop_cost = self._evaluate_candidate(
                original_image=original_image,
                individual=proposal,
                image=image,
                cost_function=cost_function,
                active_patches=active_patches,
                shape_dict=shape_dict,
                clean_ref_boxes=clean_ref_boxes,
                eps=eps,
                rgb_mode=rgb_mode,
                topk=topk,
                match_iou_thresh=match_iou_thresh,
                raw_cost_weight=raw_cost_weight,
                topk_score_weight=topk_score_weight,
                matched_score_weight=matched_score_weight,
                matched_box_weight=matched_box_weight,
                matched_iou_weight=matched_iou_weight,
                global_score_weight=global_score_weight,
                global_box_weight=global_box_weight,
                mag_weight=mag_weight
            )

            if prop_fitness > current_fitness:
                theta = np.copy(proposal)
                current_fitness = prop_fitness
                current_adv = np.copy(prop_adv)
                current_cost = prop_cost
                refine_step = min(refine_step * 1.05, 0.30)
            else:
                refine_step = max(refine_step * 0.65, 0.03)

            print(
                f"[PSO-LOCAL] step:{step_idx} "
                f"best_cost:{current_cost:.6f} "
                f"best_fitness:{current_fitness:.6f} "
                f"refine_step:{refine_step:.4f}"
            )

        return theta, float(current_fitness), current_adv, float(current_cost)

    # -------------------------
    # GA
    # -------------------------
    def _tournament_select(self, population, fitnesses, k=3):
        idxs = np.random.choice(len(population), size=k, replace=False)
        best_idx = idxs[np.argmax([fitnesses[i] for i in idxs])]
        return np.copy(population[best_idx])

    def _crossover(self, p1, p2):
        child1 = np.copy(p1)
        child2 = np.copy(p2)

        mask = np.random.rand(len(p1)) < 0.5
        child1[mask] = p2[mask]
        child2[mask] = p1[mask]
        return child1, child2

    def _mutate(self, individual, mutation_rate=0.2, eps=0.08):
        child = np.copy(individual)
        for i in range(len(child)):
            if np.random.rand() < mutation_rate:
                child[i] += np.random.normal(0, 0.02)
        return np.clip(child, -eps, eps)

    def run_ga_attack(
        self,
        original_image,
        image,
        cost_function,
        selected_boxes,
        pop_size=12,
        generations=15,
        elite_size=2,
        mutation_rate=0.2,
        grid_size=4,
        eps=0.08
    ):
        num_boxes = len(selected_boxes)
        if num_boxes == 0:
            print("[GA] No selected boxes found. Returning original image.")
            return np.copy(original_image), 0.0, 0.0

        population = [
            self._random_individual(num_boxes=num_boxes, grid_size=grid_size, eps=eps)
            for _ in range(pop_size)
        ]

        best_adv = np.copy(original_image)
        best_fitness = -1e18
        best_cost = 1e18

        best_cost_prev = float("inf")
        no_improve_count = 0
        start_time = timer()

        for gen in range(generations):
            fitnesses = []

            for individual in population:
                fitness, adv, cost = self._evaluate_candidate(
                    original_image=original_image,
                    individual=individual,
                    image=image,
                    cost_function=cost_function,
                    selected_boxes=selected_boxes,
                    grid_size=grid_size,
                    eps=eps,
                    rgb_mode=False,
                    matched_iou_weight=0.0
                )

                fitnesses.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_adv = np.copy(adv)
                    best_cost = cost

            print(f"[GA] generation:{gen} best_cost:{best_cost:.6f} best_fitness:{best_fitness:.6f}")

            if best_cost_prev - best_cost < 1e-3:
                no_improve_count += 1
            else:
                no_improve_count = 0
            best_cost_prev = best_cost

            if no_improve_count >= 3:
                print("[GA] Early stopping at generation", gen)
                break

            sorted_idx = np.argsort(fitnesses)[::-1]
            new_population = [np.copy(population[i]) for i in sorted_idx[:elite_size]]

            while len(new_population) < pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1, mutation_rate=mutation_rate, eps=eps)
                c2 = self._mutate(c2, mutation_rate=mutation_rate, eps=eps)
                new_population.append(c1)
                if len(new_population) < pop_size:
                    new_population.append(c2)

            population = new_population

        runtime_sec = timer() - start_time
        return best_adv, float(best_cost), float(runtime_sec)

    # -------------------------
    # PSO + RGB + NES + local refinement
    # -------------------------
    def run_pso_attack(
        self,
        original_image,
        image,
        cost_function,
        selected_boxes,
        swarm_size=20,
        iterations=35,
        grid_size=None,      # None => adaptive
        eps=0.08,
        rgb_mode=True,
        w_inertia=0.85,
        c1=2.0,
        c2=1.0,
        v_max_ratio=0.30,
        stagnation_limit=5,
        severe_stagnation_limit=8,
        reinit_noise_std=0.02,
        nes_refine_interval=4,
        nes_samples=6,
        nes_sigma=0.10,
        nes_step=0.22,
        nes_velocity_scale=0.12,
        final_local_refine=True,
        final_refine_steps=12,
        final_refine_samples=10,
        final_refine_sigma=0.05,
        final_refine_step=0.18,
        topk=5,
        match_iou_thresh=0.10,
        raw_cost_weight=0.10,
        topk_score_weight=1.20,
        matched_score_weight=1.50,
        matched_box_weight=0.90,
        matched_iou_weight=1.10,
        global_score_weight=0.20,
        global_box_weight=0.10,
        mag_weight=0.12
    ):
        num_boxes = len(selected_boxes)
        if num_boxes == 0:
            print("[PSO] No selected boxes found. Returning original image.")
            return np.copy(original_image), 0.0, 0.0

        if grid_size is None:
            patch_grid = self._choose_adaptive_patch_grid(selected_boxes, original_image)
        else:
            patch_grid = int(grid_size)

        active_patches = self._build_overlap_patch_specs(
            selected_boxes=selected_boxes,
            original_image=original_image,
            patch_grid=patch_grid
        )

        if len(active_patches) == 0:
            print("[PSO] No active overlap patches found. Returning original image.")
            return np.copy(original_image), 0.0, 0.0

        shape_dict, shape_names = self._build_shape_dictionary(active_patches)
        num_shapes = shape_dict.shape[0]

        if num_shapes == 0:
            print("[PSO] No shapes available. Returning original image.")
            return np.copy(original_image), 0.0, 0.0

        coeff_dim = num_shapes * 3 if rgb_mode else num_shapes
        v_max = v_max_ratio * eps

        particles = self._structured_shape_initialization(
            shape_dict=shape_dict,
            shape_names=shape_names,
            swarm_size=swarm_size,
            eps=eps,
            rgb_mode=rgb_mode
        )
        velocities = np.zeros((swarm_size, coeff_dim), dtype=np.float32)

        personal_best = np.copy(particles)
        personal_best_fitness = np.full((swarm_size,), -1e18, dtype=np.float32)
        stagnation_counter = np.zeros((swarm_size,), dtype=np.int32)

        global_best = np.copy(particles[0])
        global_best_fitness = -1e18
        global_best_adv = np.copy(original_image)
        global_best_cost = 1e18

        best_cost_prev = float("inf")
        no_improve_count = 0
        start_time = timer()

        clean_ref_boxes = selected_boxes
        nes_grad = np.zeros(coeff_dim, dtype=np.float32)

        for it in range(iterations):
            progress = it / max(1, iterations - 1)

            w = 0.85 - 0.40 * progress
            c1_t = 2.0 - 0.8 * progress
            c2_t = 1.0 + 0.8 * progress

            for i in range(swarm_size):
                fitness, adv, cost = self._evaluate_candidate(
                    original_image=original_image,
                    individual=particles[i],
                    image=image,
                    cost_function=cost_function,
                    active_patches=active_patches,
                    shape_dict=shape_dict,
                    clean_ref_boxes=clean_ref_boxes,
                    eps=eps,
                    rgb_mode=rgb_mode,
                    topk=topk,
                    match_iou_thresh=match_iou_thresh,
                    raw_cost_weight=raw_cost_weight,
                    topk_score_weight=topk_score_weight,
                    matched_score_weight=matched_score_weight,
                    matched_box_weight=matched_box_weight,
                    matched_iou_weight=matched_iou_weight,
                    global_score_weight=global_score_weight,
                    global_box_weight=global_box_weight,
                    mag_weight=mag_weight
                )

                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best[i] = np.copy(particles[i])
                    stagnation_counter[i] = 0
                else:
                    stagnation_counter[i] += 1

                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best = np.copy(particles[i])
                    global_best_adv = np.copy(adv)
                    global_best_cost = cost

            if (it % nes_refine_interval == 0) or (it == iterations - 1):
                nes_grad = self._estimate_nes_gradient(
                    theta=global_best,
                    original_image=original_image,
                    image=image,
                    cost_function=cost_function,
                    active_patches=active_patches,
                    shape_dict=shape_dict,
                    clean_ref_boxes=clean_ref_boxes,
                    eps=eps,
                    rgb_mode=rgb_mode,
                    nes_sigma=nes_sigma,
                    nes_samples=nes_samples,
                    topk=topk,
                    match_iou_thresh=match_iou_thresh,
                    raw_cost_weight=raw_cost_weight,
                    topk_score_weight=topk_score_weight,
                    matched_score_weight=matched_score_weight,
                    matched_box_weight=matched_box_weight,
                    matched_iou_weight=matched_iou_weight,
                    global_score_weight=global_score_weight,
                    global_box_weight=global_box_weight,
                    mag_weight=mag_weight
                )

                refined = np.clip(global_best + nes_step * nes_grad, -eps, eps)

                refined_fitness, refined_adv, refined_cost = self._evaluate_candidate(
                    original_image=original_image,
                    individual=refined,
                    image=image,
                    cost_function=cost_function,
                    active_patches=active_patches,
                    shape_dict=shape_dict,
                    clean_ref_boxes=clean_ref_boxes,
                    eps=eps,
                    rgb_mode=rgb_mode,
                    topk=topk,
                    match_iou_thresh=match_iou_thresh,
                    raw_cost_weight=raw_cost_weight,
                    topk_score_weight=topk_score_weight,
                    matched_score_weight=matched_score_weight,
                    matched_box_weight=matched_box_weight,
                    matched_iou_weight=matched_iou_weight,
                    global_score_weight=global_score_weight,
                    global_box_weight=global_box_weight,
                    mag_weight=mag_weight
                )

                if refined_fitness > global_best_fitness:
                    global_best_fitness = refined_fitness
                    global_best = np.copy(refined)
                    global_best_adv = np.copy(refined_adv)
                    global_best_cost = refined_cost

            print(
                f"[PSO-RGB-NES] iter:{it} "
                f"best_cost:{global_best_cost:.6f} "
                f"best_fitness:{global_best_fitness:.6f} "
                f"grid:{patch_grid} "
                f"num_active_patches:{len(active_patches)} "
                f"num_shapes:{num_shapes} "
                f"coeff_dim:{coeff_dim} "
                f"w:{w:.3f} c1:{c1_t:.3f} c2:{c2_t:.3f}"
            )

            if best_cost_prev - global_best_cost < 1e-4:
                no_improve_count += 1
            else:
                no_improve_count = 0
            best_cost_prev = global_best_cost

            if no_improve_count >= 7:
                print("[PSO-RGB-NES] Early stopping at iteration", it)
                break

            for i in range(swarm_size):
                if stagnation_counter[i] >= severe_stagnation_limit:
                    new_seed = self._structured_shape_initialization(
                        shape_dict=shape_dict,
                        shape_names=shape_names,
                        swarm_size=1,
                        eps=eps,
                        rgb_mode=rgb_mode
                    )[0]
                    particles[i] = np.copy(new_seed)
                    velocities[i] = np.zeros(coeff_dim, dtype=np.float32)
                    stagnation_counter[i] = 0
                    continue

                elif stagnation_counter[i] >= stagnation_limit:
                    if np.random.rand() < 0.5:
                        particles[i] = np.clip(
                            global_best + np.random.normal(0, reinit_noise_std, size=coeff_dim),
                            -eps,
                            eps
                        ).astype(np.float32)
                    else:
                        new_seed = self._structured_shape_initialization(
                            shape_dict=shape_dict,
                            shape_names=shape_names,
                            swarm_size=1,
                            eps=eps,
                            rgb_mode=rgb_mode
                        )[0]
                        particles[i] = np.copy(new_seed)

                    velocities[i] = np.zeros(coeff_dim, dtype=np.float32)
                    stagnation_counter[i] = 0
                    continue

                r1 = np.random.rand(coeff_dim).astype(np.float32)
                r2 = np.random.rand(coeff_dim).astype(np.float32)

                velocities[i] = (
                    w * velocities[i]
                    + c1_t * r1 * (personal_best[i] - particles[i])
                    + c2_t * r2 * (global_best - particles[i])
                    + nes_velocity_scale * nes_grad
                )

                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                particles[i] = np.clip(particles[i] + velocities[i], -eps, eps)

        if final_local_refine:
            global_best, global_best_fitness, global_best_adv, global_best_cost = self._local_refine_best_blackbox(
                best_theta=global_best,
                best_fitness=global_best_fitness,
                best_adv=global_best_adv,
                best_cost=global_best_cost,
                original_image=original_image,
                image=image,
                cost_function=cost_function,
                active_patches=active_patches,
                shape_dict=shape_dict,
                clean_ref_boxes=clean_ref_boxes,
                eps=eps,
                rgb_mode=rgb_mode,
                refine_steps=final_refine_steps,
                refine_sigma=final_refine_sigma,
                refine_samples=final_refine_samples,
                refine_step=final_refine_step,
                topk=topk,
                match_iou_thresh=match_iou_thresh,
                raw_cost_weight=raw_cost_weight,
                topk_score_weight=topk_score_weight,
                matched_score_weight=matched_score_weight,
                matched_box_weight=matched_box_weight,
                matched_iou_weight=matched_iou_weight,
                global_score_weight=global_score_weight,
                global_box_weight=global_box_weight,
                mag_weight=mag_weight
            )

        runtime_sec = timer() - start_time
        return global_best_adv, float(global_best_cost), float(runtime_sec)

    # -------------------------
    # DE
    # -------------------------
    def run_de_attack(
        self,
        original_image,
        image,
        cost_function,
        selected_boxes,
        pop_size=12,
        generations=15,
        grid_size=4,
        eps=0.08,
        F=0.5,
        CR=0.9
    ):
        num_boxes = len(selected_boxes)
        if num_boxes == 0:
            print("[DE] No selected boxes found. Returning original image.")
            return np.copy(original_image), 0.0, 0.0

        dim = num_boxes * grid_size * grid_size

        population = np.random.uniform(-eps, eps, size=(pop_size, dim)).astype(np.float32)
        fitnesses = np.zeros((pop_size,), dtype=np.float32)

        best_adv = np.copy(original_image)
        best_fitness = -1e18
        best_cost = 1e18

        start_time = timer()

        for i in range(pop_size):
            fitness, adv, cost = self._evaluate_candidate(
                original_image=original_image,
                individual=population[i],
                image=image,
                cost_function=cost_function,
                selected_boxes=selected_boxes,
                grid_size=grid_size,
                eps=eps,
                rgb_mode=False,
                matched_iou_weight=0.0
            )
            fitnesses[i] = fitness

            if fitness > best_fitness:
                best_fitness = fitness
                best_adv = np.copy(adv)
                best_cost = cost

        best_cost_prev = float("inf")
        no_improve_count = 0

        for gen in range(generations):
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)

                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, -eps, eps)

                trial = np.copy(population[i])
                j_rand = np.random.randint(dim)

                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                trial = np.clip(trial, -eps, eps)

                trial_fitness, trial_adv, trial_cost = self._evaluate_candidate(
                    original_image=original_image,
                    individual=trial,
                    image=image,
                    cost_function=cost_function,
                    selected_boxes=selected_boxes,
                    grid_size=grid_size,
                    eps=eps,
                    rgb_mode=False,
                    matched_iou_weight=0.0
                )

                if trial_fitness > fitnesses[i]:
                    population[i] = np.copy(trial)
                    fitnesses[i] = trial_fitness

                    if trial_fitness > best_fitness:
                        best_fitness = trial_fitness
                        best_adv = np.copy(trial_adv)
                        best_cost = trial_cost

            print(f"[DE] generation:{gen} best_cost:{best_cost:.6f} best_fitness:{best_fitness:.6f}")

            if best_cost_prev - best_cost < 1e-3:
                no_improve_count += 1
            else:
                no_improve_count = 0
            best_cost_prev = best_cost

            if no_improve_count >= 3:
                print("[DE] Early stopping at generation", gen)
                break

        runtime_sec = timer() - start_time
        return best_adv, float(best_cost), float(runtime_sec)