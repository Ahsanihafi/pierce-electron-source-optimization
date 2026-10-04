from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import time

waktu0 = time.time()

"""
================================================================================
MOEA/D-DE using three independently trained Gaussian-process surrogate models

Design variables:
    Seven variables. Internally, every gene is represented in the normalized
    interval [0, 1], using the X_scaler saved during GP training.

Physical GP outputs:
    1. Current     (negative in the CST output)
    2. Emittance
    3. Envelope

MOEA/D objectives:
    All objectives are minimized.

    1. Scaled current
       Because the CST current is negative, minimizing it favors a more
       negative current and therefore a larger current magnitude.
    2. Scaled emittance
    3. Scaled envelope

The uncertainty-penalized scaled GP predictions are used for MOEA/D aggregation:

    robust objective = scaled GP mean + eta * scaled GP standard deviation

The standard deviation is the latent-function posterior standard deviation, so
it represents GP epistemic uncertainty and does not include the fitted noise
variance sigma_n^2. Physical predictions are calculated only after MOEA/D has
finished, when the final results are saved.
================================================================================
"""


# ==============================================================================
# 1. Gaussian-process loading and inference
# ==============================================================================

def load_gp_models(file_path):
    """Load the three trained GP models from one pickle file."""

    with open(file_path, "rb") as file:
        all_models = pickle.load(file)

    required_models = ["current", "emittance", "envelope"]

    for model_name in required_models:
        if model_name not in all_models:
            raise KeyError(
                f"The pickle file does not contain the '{model_name}' model."
            )

    return all_models


def check_gp_models(all_models):
    """
    Check that all three GPs use the same seven-dimensional input scaling.

    The three models were trained using the same X_train data, so their fitted
    input scalers should be identical.
    """

    model_current = all_models["current"]
    model_emittance = all_models["emittance"]
    model_envelope = all_models["envelope"]

    models = [model_current, model_emittance, model_envelope]
    reference_scaler = model_current["X_scaler"]

    if reference_scaler.n_features_in_ != 7:
        raise ValueError(
            "The GP input scaler must contain exactly 7 input variables."
        )

    for model in models[1:]:
        scaler = model["X_scaler"]

        if not np.allclose(scaler.data_min_, reference_scaler.data_min_):
            raise ValueError("The GP models use different input minima.")

        if not np.allclose(scaler.data_max_, reference_scaler.data_max_):
            raise ValueError("The GP models use different input maxima.")


def rbf_kernel(X1, X2, length_scales, sigma_f):
    """
    Calculate the ARD RBF covariance between two collections of points.

    Parameters
    ----------
    X1 : ndarray, shape (N1, D)
    X2 : ndarray, shape (N2, D)

    Returns
    -------
    K : ndarray, shape (N1, N2)

    Notes
    -----
    This is the vectorized version of the RBF kernel used during training.
    It follows exactly the same equation but is much faster during MOEA/D,
    where many candidate designs must be evaluated.
    """

    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    length_scales = np.asarray(length_scales, dtype=float)

    if X1.ndim != 2 or X2.ndim != 2:
        raise ValueError("X1 and X2 must both be two-dimensional arrays.")

    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 contain different numbers of variables.")

    # Shape after broadcasting: (N1, N2, D)
    normalized_difference = (
        X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    ) / length_scales[np.newaxis, np.newaxis, :]

    # Sum the squared normalized differences over the D input variables.
    squared_distance = np.sum(
        normalized_difference**2,
        axis=2
    )

    K = sigma_f**2 * np.exp(
        -0.5 * squared_distance
    )

    return K


def predict_gp_mean_std_scaled(model, X_star_scaled):
    """
    Predict one GP output for a batch of normalized candidate designs.

    Returns
    -------
    mean_scaled : ndarray, shape (number_of_candidates,)
        GP mean in scaled output units.

    std_scaled : ndarray, shape (number_of_candidates,)
        GP epistemic standard deviation in scaled output units.
    """

    X_train = model["X_train"]
    length_scales = model["length_scales"]
    sigma_f = model["sigma_f"]
    L = model["L"]
    alpha = model["alpha"]

    # K_star has shape:
    # number of training samples x number of candidate designs
    K_star = rbf_kernel(
        X_train,
        X_star_scaled,
        length_scales,
        sigma_f
    )

    # For each candidate:
    # mean = k_star^T alpha
    mean_scaled = K_star.T @ alpha

    # Solve L V = K_star. Each column corresponds to one candidate.
    V = np.linalg.solve(L, K_star)

    # Latent-function posterior variance. sigma_n^2 is deliberately not
    # added because the final optimization penalizes epistemic uncertainty.
    variance_scaled = sigma_f**2 - np.sum(V**2, axis=0)

    # Prevent small negative values caused by floating-point rounding.
    variance_scaled = np.maximum(variance_scaled, 0.0)
    std_scaled = np.sqrt(variance_scaled)

    return (
        np.asarray(mean_scaled).reshape(-1),
        np.asarray(std_scaled).reshape(-1)
    )


def inverse_scale_gp_results(mean_scaled, std_scaled, eta, model):
    """
    Convert final scaled GP results to physical output units.

    For log-transformed outputs, the nominal physical prediction is exp(mu_log)
    and the physical uncertainty follows the corresponding lognormal
    distribution. The robust physical value is the inverse transformation of
    mu_scaled + eta * sigma_scaled.
    """

    mean_scaled_2d = np.asarray(
        mean_scaled,
        dtype=float
    ).reshape(-1, 1)

    y_scaler = model["y_scaler"]
    mean_inverse_scaled = y_scaler.inverse_transform(mean_scaled_2d).reshape(-1)

    std_scaled = np.asarray(
        std_scaled,
        dtype=float
    ).reshape(-1)

    # sklearn uses y_scaled = scale * y_transformed + offset.
    std_inverse_scaled = std_scaled / abs(y_scaler.scale_[0])

    robust_scaled = mean_scaled_2d.reshape(-1) + eta * std_scaled
    robust_inverse_scaled = y_scaler.inverse_transform(
        robust_scaled.reshape(-1, 1)
    ).reshape(-1)

    output_transform = model.get("output_transform", "none")

    if output_transform == "log":
        # Convert log(y) back to y.
        mean_physical = np.exp(mean_inverse_scaled)
        std_physical = (
            np.exp(
                mean_inverse_scaled
                + 0.5 * std_inverse_scaled**2
            )
            * np.sqrt(
                np.exp(std_inverse_scaled**2) - 1.0
            )
        )
        robust_physical = np.exp(robust_inverse_scaled)
    elif output_transform == "none":
        mean_physical = mean_inverse_scaled
        std_physical = std_inverse_scaled
        robust_physical = robust_inverse_scaled
    else:
        raise ValueError("Unknown output transformation.")

    return mean_physical, std_physical, robust_physical

def batch_gp_inferencing(
    total_pop,
    batch_size,
    all_models,
    eta
):
    """
    Evaluate all candidates using the three GP models.

    The genes stored in each individual are already normalized to [0, 1].

    Returns
    -------
    objective_scaled : ndarray, shape (population_size, 3)
        Values of mean_scaled + eta * std_scaled used by MOEA/D.

    mean_scaled : ndarray, shape (population_size, 3)
        Nominal GP means in scaled output units.

    std_scaled : ndarray, shape (population_size, 3)
        GP epistemic standard deviations in scaled output units.
    """

    if eta < 0.0:
        raise ValueError("eta must be non-negative.")

    objective_scaled_batches = []
    mean_scaled_batches = []
    std_scaled_batches = []

    pop_size = len(total_pop)

    for start in range(0, pop_size, batch_size):
        current_batch = total_pop[
            start:start + batch_size
        ]

        # Shape: batch size x 7
        X_batch_scaled = np.vstack([
            individual.gene
            for individual in current_batch
        ])

        objective_columns = []
        mean_columns = []
        std_columns = []

        for model_name in ["current", "emittance", "envelope"]:
            model = all_models[model_name]

            mean_scaled, std_scaled = predict_gp_mean_std_scaled(
                model,
                X_batch_scaled
            )

            robust_scaled = mean_scaled + eta * std_scaled

            objective_columns.append(robust_scaled)
            mean_columns.append(mean_scaled)
            std_columns.append(std_scaled)

        objective_scaled_batches.append(
            np.column_stack(objective_columns)
        )

        mean_scaled_batches.append(
            np.column_stack(mean_columns)
        )

        std_scaled_batches.append(
            np.column_stack(std_columns)
        )

    objective_scaled = np.vstack(
        objective_scaled_batches
    )

    mean_scaled = np.vstack(
        mean_scaled_batches
    )

    std_scaled = np.vstack(
        std_scaled_batches
    )

    return objective_scaled, mean_scaled, std_scaled


# ==============================================================================
# 2. Individual used by MOEA/D
# ==============================================================================

class Individual:
    def __init__(self, length_input, length_output, identity):
        # Normalized design variables in [0, 1]
        self.gene = np.zeros(length_input)

        # Scaled objectives used by MOEA/D
        self.trait = np.zeros(length_output)

        # Nominal scaled GP means and scaled epistemic standard deviations.
        # These are retained so that physical results can be calculated once,
        # after the optimization has finished.
        self.mean_scaled = np.zeros(length_output)
        self.std_scaled = np.zeros(length_output)

        self.weight = np.zeros(length_output)
        self.identity = identity
        self.aggregation = 0.0
        self.rank = 0

    def transfer(self, other: Individual):
        """Replace this individual with a copy of another individual."""

        self.gene = other.gene.copy()
        self.trait = other.trait.copy()
        self.mean_scaled = other.mean_scaled.copy()
        self.std_scaled = other.std_scaled.copy()
        self.aggregation = other.aggregation
        self.rank = other.rank


def objective_function(
    population,
    batch_size,
    all_models,
    eta
):
    """Evaluate a population and assign its GP predictions."""

    objective_scaled, mean_scaled, std_scaled = batch_gp_inferencing(
        population,
        batch_size,
        all_models,
        eta
    )

    for i in range(len(population)):
        population[i].trait = objective_scaled[i].copy()
        population[i].mean_scaled = mean_scaled[i].copy()
        population[i].std_scaled = std_scaled[i].copy()


# ==============================================================================
# 3. Das-Dennis weight generation
# ==============================================================================

def generator_das(length_output, order, current=None, results=None):
    if current is None:
        current = []

    if results is None:
        results = []

    if len(current) == length_output - 1:
        leftover = order - sum(current)
        results.append(current + [leftover])
        return results

    for value in range(order - sum(current) + 1):
        generator_das(
            length_output,
            order,
            current + [value],
            results
        )

    return results


def das_dennis(length_output, order, pop_size):
    weight = np.array(
        generator_das(length_output, order),
        dtype=float
    ) / order

    if pop_size != len(weight):
        raise ValueError(
            f"pop_size {pop_size} != number of weight vectors {len(weight)}. "
            "Adjust order_das or pop_size."
        )

    print("Weight vectors have been generated.")
    return weight


# ==============================================================================
# 4. Neighborhood calculation
# ==============================================================================

def compute_neighbor(population, num_neighbor):
    pop_size = len(population)

    weight_matrix = np.vstack([
        individual.weight
        for individual in population
    ])

    neighborhood = []

    for i in range(pop_size):
        difference = weight_matrix - weight_matrix[i]
        distance = np.sqrt(
            np.sum(difference**2, axis=1)
        )

        sorted_indices = np.argsort(distance)

        # The MOEA/D replacement neighborhood includes the subproblem itself.
        # The target will be excluded separately when selecting DE parents.
        neighborhood.append(
            sorted_indices[:num_neighbor]
        )

    return neighborhood


# ==============================================================================
# 5. Differential-evolution operations
# ==============================================================================

def mutation(parent_indices, population, mutation_rate):
    r1, r2, r3 = parent_indices

    x1 = population[r1].gene
    x2 = population[r2].gene
    x3 = population[r3].gene

    return x1 + mutation_rate * (x2 - x3)


def binomial_crossover(parent_gene, mutant_gene, crossover_rate):
    dimension = len(parent_gene)
    trial_gene = np.zeros(dimension)

    # At least one component must come from the mutant.
    j_random = np.random.randint(dimension)

    for j in range(dimension):
        if np.random.rand() < crossover_rate or j == j_random:
            trial_gene[j] = mutant_gene[j]
        else:
            trial_gene[j] = parent_gene[j]

    return trial_gene


def repair_resample(gene, lower_bound, upper_bound):
    """Resample any component that leaves its permitted interval."""

    repaired_gene = gene.copy()

    for j in range(len(repaired_gene)):
        if (
            repaired_gene[j] < lower_bound[j]
            or repaired_gene[j] > upper_bound[j]
        ):
            repaired_gene[j] = np.random.uniform(
                lower_bound[j],
                upper_bound[j]
            )

    return repaired_gene


# ==============================================================================
# 6. Tchebycheff aggregation
# ==============================================================================

def aggregation(objective_values, weight, ideal_point):
    difference = np.abs(
        objective_values - ideal_point
    )

    # A zero weight is replaced by a very small number so that an objective is
    # not completely ignored at the extreme Das-Dennis weight vectors.
    effective_weight = np.maximum(weight, 1.0e-6)

    weighted_difference = effective_weight * difference

    return np.max(weighted_difference)


def population_aggregation(population, ideal_point):
    for individual in population:
        individual.aggregation = aggregation(
            individual.trait,
            individual.weight,
            ideal_point
        )


# ==============================================================================
# 7. Population initialization and ideal point
# ==============================================================================

def initialize_population(
    pop_size,
    length_input,
    length_output,
    lower_bound,
    upper_bound,
    order_das,
    batch_size,
    all_models,
    eta
):
    population = []

    weight = das_dennis(
        length_output,
        order_das,
        pop_size
    )

    for i in range(pop_size):
        individual = Individual(
            length_input,
            length_output,
            i
        )

        individual.weight = weight[i].copy()

        individual.gene = np.random.uniform(
            lower_bound,
            upper_bound
        )

        population.append(individual)

    objective_function(
        population,
        batch_size,
        all_models,
        eta
    )

    return population


def compute_ideal_point(population):
    objective_matrix = np.vstack([
        individual.trait
        for individual in population
    ])

    return np.min(
        objective_matrix,
        axis=0
    )


# ==============================================================================
# 8. Dominance and non-dominated sorting
# ==============================================================================

def dominates(individual_a, individual_b):
    objective_a = individual_a.trait
    objective_b = individual_b.trait

    less_or_equal = np.all(
        objective_a <= objective_b
    )

    strictly_less = np.any(
        objective_a < objective_b
    )

    return less_or_equal and strictly_less


def non_dominated_sort(population):
    dominated_set = {}
    domination_count = {}
    fronts = [[]]

    for individual_p in population:
        dominated_set[individual_p] = []
        domination_count[individual_p] = 0

        for individual_q in population:
            if individual_p is individual_q:
                continue

            if dominates(individual_p, individual_q):
                dominated_set[individual_p].append(individual_q)
            elif dominates(individual_q, individual_p):
                domination_count[individual_p] += 1

        if domination_count[individual_p] == 0:
            individual_p.rank = 0
            fronts[0].append(individual_p)

    front_index = 0

    while len(fronts[front_index]) > 0:
        next_front = []

        for individual_p in fronts[front_index]:
            for individual_q in dominated_set[individual_p]:
                domination_count[individual_q] -= 1

                if domination_count[individual_q] == 0:
                    individual_q.rank = front_index + 1
                    next_front.append(individual_q)

        front_index += 1
        fronts.append(next_front)

    if len(fronts[-1]) == 0:
        fronts.pop()

    return fronts


# ==============================================================================
# 9. Main MOEA/D-DE evolution
# ==============================================================================

def evolve(
    pop_size,
    number_of_generations,
    mutation_rate,
    crossover_rate,
    length_input,
    lower_bound,
    upper_bound,
    order_das,
    batch_size,
    all_models,
    eta,
    num_neighbor=None,
    maximum_replacements=None
):
    length_output = 3

    population = initialize_population(
        pop_size,
        length_input,
        length_output,
        lower_bound,
        upper_bound,
        order_das,
        batch_size,
        all_models,
        eta
    )

    ideal_point = compute_ideal_point(population)

    if num_neighbor is None:
        num_neighbor = max(
            10,
            pop_size // 10
        )

    num_neighbor = min(
        num_neighbor,
        pop_size - 1
    )

    if num_neighbor < 4:
        raise ValueError(
            "num_neighbor must be at least 4 so that the neighborhood "
            "contains the target and three distinct DE parents."
        )

    if maximum_replacements is None:
        maximum_replacements = max(
            1,
            num_neighbor // 10
        )

    neighborhood = compute_neighbor(
        population,
        num_neighbor
    )

    print("Number of neighbors:", num_neighbor)
    print("Maximum replacements per offspring:", maximum_replacements)

    for generation in range(number_of_generations):
        offspring = []

        for i in range(pop_size):
            neighbor_indices = neighborhood[i]

            # DE/rand/1 uses three distinct parents. Exclude the target
            # subproblem itself from the parent pool.
            parent_pool = neighbor_indices[
                neighbor_indices != i
            ]

            parent_indices = np.random.choice(
                parent_pool,
                3,
                replace=False
            )

            mutant_gene = mutation(
                parent_indices,
                population,
                mutation_rate
            )

            mutant_gene = repair_resample(
                mutant_gene,
                lower_bound,
                upper_bound
            )

            trial_gene = binomial_crossover(
                population[i].gene,
                mutant_gene,
                crossover_rate
            )

            trial_gene = repair_resample(
                trial_gene,
                lower_bound,
                upper_bound
            )

            daughter = Individual(
                length_input,
                length_output,
                i
            )

            daughter.gene = trial_gene
            daughter.weight = population[i].weight.copy()

            offspring.append(daughter)

        # One batched GP evaluation for all offspring.
        objective_function(
            offspring,
            batch_size,
            all_models,
            eta
        )

        offspring_ideal = compute_ideal_point(
            offspring
        )

        ideal_point = np.minimum(
            ideal_point,
            offspring_ideal
        )

        # Compare each offspring with the neighboring subproblems.
        for i in range(pop_size):
            replacement_count = 0

            # Random order avoids a systematic preference for early neighbors.
            neighbor_indices = np.random.permutation(
                neighborhood[i]
            )

            for neighbor_index in neighbor_indices:
                neighbor_weight = population[neighbor_index].weight

                old_aggregation = aggregation(
                    population[neighbor_index].trait,
                    neighbor_weight,
                    ideal_point
                )

                # The offspring must be evaluated using the weight of the
                # neighboring subproblem being considered for replacement.
                new_aggregation = aggregation(
                    offspring[i].trait,
                    neighbor_weight,
                    ideal_point
                )

                if new_aggregation < old_aggregation:
                    population[neighbor_index].transfer(
                        offspring[i]
                    )

                    population[neighbor_index].weight = (
                        neighbor_weight.copy()
                    )

                    population[neighbor_index].aggregation = new_aggregation

                    replacement_count += 1

                    if replacement_count >= maximum_replacements:
                        break

        if generation % 10 == 0 or generation == number_of_generations - 1:
            print(
                f"Generation {generation + 1}/{number_of_generations}, "
                f"ideal point = {ideal_point}"
            )

    population_aggregation(
        population,
        ideal_point
    )

    return population


# ==============================================================================
# 10. Result processing
# ==============================================================================

def deduplicate_population(population, decimals=8):
    unique_population = []
    seen = set()

    for individual in population:
        key = tuple(
            np.round(individual.gene, decimals=decimals)
        )

        if key not in seen:
            seen.add(key)
            unique_population.append(individual)

    return unique_population


def population_to_array(
    population,
    all_models,
    eta
):
    """
    Convert the final normalized designs and GP results to reportable values.

    The inverse output transformations are performed here, after MOEA/D has
    finished, rather than during every objective-function evaluation.
    """

    X_scaled = np.vstack([
        individual.gene
        for individual in population
    ])

    X_scaler = all_models["current"]["X_scaler"]

    X_physical = X_scaler.inverse_transform(
        X_scaled
    )

    mean_scaled = np.vstack([
        individual.mean_scaled
        for individual in population
    ])

    std_scaled = np.vstack([
        individual.std_scaled
        for individual in population
    ])

    robust_scaled = np.vstack([
        individual.trait
        for individual in population
    ])

    physical_columns = []
    scaled_columns = []

    model_names = ["current", "emittance", "envelope"]

    for output_index, model_name in enumerate(model_names):
        model = all_models[model_name]

        mean_physical, std_physical, robust_physical = (
            inverse_scale_gp_results(
                mean_scaled[:, output_index],
                std_scaled[:, output_index],
                eta,
                model
            )
        )

        physical_columns.extend([
            mean_physical,
            std_physical,
            robust_physical
        ])

        scaled_columns.extend([
            mean_scaled[:, output_index],
            std_scaled[:, output_index],
            robust_scaled[:, output_index]
        ])

    return np.column_stack([
        X_physical,
        *physical_columns,
        *scaled_columns
    ])


def save_optimization_results(
    population,
    all_models,
    output_directory,
    eta
):
    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    unique_population = deduplicate_population(
        population
    )

    fronts = non_dominated_sort(
        unique_population
    )

    pareto_front = fronts[0]
    population_array = population_to_array(
        unique_population,
        all_models,
        eta
    )

    pareto_array = population_to_array(
        pareto_front,
        all_models,
        eta
    )

    header = (
        "x1,x2,x3,x4,x5,x6,x7,"
        "current_predicted,current_std,current_robust,"
        "emittance_predicted,emittance_std,emittance_robust,"
        "envelope_predicted,envelope_std,envelope_robust,"
        "current_mean_scaled,current_std_scaled,current_robust_scaled,"
        "emittance_mean_scaled,emittance_std_scaled,emittance_robust_scaled,"
        "envelope_mean_scaled,envelope_std_scaled,envelope_robust_scaled"
    )

    population_file = output_directory / "moead_final_population_robust_final.csv"
    pareto_file = output_directory / "moead_pareto_front_robust_final.csv"

    population_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savetxt(
        population_file,
        population_array,
        delimiter=",",
        header=header,
        comments=""
    )

    np.savetxt(
        pareto_file,
        pareto_array,
        delimiter=",",
        header=header,
        comments=""
    )

    print("Final population saved to:", population_file)
    print("Pareto front saved to:", pareto_file)
    print("Number of unique final solutions:", len(unique_population))
    print("Number of non-dominated solutions:", len(pareto_front))

    return pareto_front


# ==============================================================================
# 11. Execute the optimization
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(12345)

    model_file = "trained_models/electron_source_gp_retrain2.pkl"
    output_directory = "optimization_results/robust"

    all_models = load_gp_models(
        model_file
    )

    check_gp_models(
        all_models
    )

    # Every gene is stored in normalized GP input coordinates.
    length_input = 7
    lower_bound = np.zeros(length_input)
    upper_bound = np.ones(length_input)

    # For 3 objectives, Das-Dennis order H gives:
    # population size = (H + 2)(H + 1) / 2.
    # H = 12 gives 91 weight vectors and therefore 91 individuals.
    order_das = 12
    pop_size = (
        (order_das + 2) * (order_das + 1)
    ) // 2

    number_of_generations = 500
    mutation_rate = 0.5
    crossover_rate = 0.9
    batch_size = pop_size
    num_neighbor = 10
    maximum_replacements = 2

    # Epistemic-uncertainty penalty used for all three minimized objectives:
    # robust objective = mean_scaled + eta * std_scaled
    eta = 3.0

    final_population = evolve(
        pop_size=pop_size,
        number_of_generations=number_of_generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        length_input=length_input,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        order_das=order_das,
        batch_size=batch_size,
        all_models=all_models,
        eta=eta,
        num_neighbor=num_neighbor,
        maximum_replacements=maximum_replacements
    )

    save_optimization_results(
        final_population,
        all_models,
        output_directory,
        eta
    )

waktuf = time.time()
print("Durasi:", waktuf-waktu0)