import numpy as np
from timeit import default_timer as timer
from keras import backend as K


class MetaheuristicAttacks:
    def _random_individual(self, num_boxes, grid_size=4, eps=0.08):
        num_vars = num_boxes * grid_size * grid_size
        return np.random.uniform(-eps, eps, size=(num_vars,)).astype(np.float32)

    def _evaluate_candidate(
        self,
        original_image,
        individual,
        image,
        cost_function,
        selected_boxes,
        grid_size=4,
        score_weight=1.0,
        box_weight=0.3,
        mag_weight=0.2
    ):
        adv = self._apply_mask_from_individual(
            original_image,
            individual,
            selected_boxes,
            grid_size=grid_size
        )

        cost = self.sess.run(
            cost_function,
            feed_dict={
                self.yolo4_model.input: adv,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            }
        )

        summary = self._get_detection_summary(adv, image)
        num_boxes = summary["num_boxes"]
        score_sum = summary["score_sum"]

        mag_penalty = np.mean(np.abs(individual))

        fitness = -(
            score_weight * float(score_sum)
            + box_weight * float(num_boxes)
            + mag_weight * float(mag_penalty)
        )

        return fitness, adv, float(cost)

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
                    grid_size=grid_size
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
    # PSO
    # -------------------------
    def run_pso_attack(
        self,
        original_image,
        image,
        cost_function,
        selected_boxes,
        swarm_size=20,
        iterations=30,
        grid_size=4,
        eps=0.08,
        w_inertia=0.9,
        c1=2.5,
        c2=0.5,
        v_max_ratio=0.2,
        stagnation_limit=5,
        reselect_interval=5,
        reinit_noise_std=0.02
    ):
        num_boxes = len(selected_boxes)
        if num_boxes == 0:
            print("[PSO] No selected boxes found. Returning original image.")
            return np.copy(original_image), 0.0, 0.0

        dim = num_boxes * grid_size * grid_size
        v_max = v_max_ratio * eps

        # -------------------------
        # Mixed initialization
        # -------------------------
        particles = np.zeros((swarm_size, dim), dtype=np.float32)

        n_zero = max(1, swarm_size // 4)
        n_gauss = max(1, swarm_size // 2)
        n_uniform = swarm_size - n_zero - n_gauss

        idx = 0

        for _ in range(n_zero):
            particles[idx] = np.random.normal(0.0, 0.005, size=dim).astype(np.float32)
            idx += 1

        for _ in range(n_gauss):
            particles[idx] = np.random.normal(0.0, eps / 4.0, size=dim).astype(np.float32)
            idx += 1

        for _ in range(n_uniform):
            particles[idx] = np.random.uniform(-eps, eps, size=(dim,)).astype(np.float32)
            idx += 1

        particles = np.clip(particles, -eps, eps)
        velocities = np.zeros((swarm_size, dim), dtype=np.float32)

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

        for it in range(iterations):
            # -------------------------
            # Dynamic box reselection
            # -------------------------
            if it > 0 and (it % reselect_interval == 0):
                summary = self._get_detection_summary(global_best_adv, image)

                # Only reselect if we still have enough boxes to preserve dimension
                if len(summary["boxes"]) >= num_boxes:
                    selected_boxes = self._select_topk_boxes_by_score(
                        summary["boxes"],
                        summary["scores"],
                        k=num_boxes
                    )
                    print(f"[PSO] Reselected top-{num_boxes} boxes at iter {it}")

            # -------------------------
            # Adaptive coefficients
            # -------------------------
            progress = it / max(1, iterations - 1)
            w = 0.9 - 0.5 * progress      # 0.9 -> 0.4
            c1_t = 2.5 - 1.5 * progress   # 2.5 -> 1.0
            c2_t = 0.5 + 1.5 * progress   # 0.5 -> 2.0

            for i in range(swarm_size):
                fitness, adv, cost = self._evaluate_candidate(
                    original_image=original_image,
                    individual=particles[i],
                    image=image,
                    cost_function=cost_function,
                    selected_boxes=selected_boxes,
                    grid_size=grid_size
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

            print(
                f"[PSO] iter:{it} "
                f"best_cost:{global_best_cost:.6f} "
                f"best_fitness:{global_best_fitness:.6f} "
                f"w:{w:.3f} c1:{c1_t:.3f} c2:{c2_t:.3f}"
            )

            if best_cost_prev - global_best_cost < 1e-4:
                no_improve_count += 1
            else:
                no_improve_count = 0
            best_cost_prev = global_best_cost

            if no_improve_count >= 6:
                print("[PSO] Early stopping at iteration", it)
                break

            for i in range(swarm_size):
                # Reinitialize stagnant particles near global best
                if stagnation_counter[i] >= stagnation_limit:
                    particles[i] = np.clip(
                        global_best + np.random.normal(0, reinit_noise_std, size=dim),
                        -eps,
                        eps
                    ).astype(np.float32)
                    velocities[i] = np.zeros(dim, dtype=np.float32)
                    stagnation_counter[i] = 0
                    continue

                r1 = np.random.rand(dim).astype(np.float32)
                r2 = np.random.rand(dim).astype(np.float32)

                velocities[i] = (
                    w * velocities[i]
                    + c1_t * r1 * (personal_best[i] - particles[i])
                    + c2_t * r2 * (global_best - particles[i])
                )

                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                particles[i] = np.clip(particles[i] + velocities[i], -eps, eps)

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
                grid_size=grid_size
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
                    grid_size=grid_size
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