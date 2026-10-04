import numpy as np
import pickle

"""
================================================================================
We need to use the same RBF kernel to do inference
so X1 and X2 is a two dimensional matrix
the first index represent the dataset
the second index represent the input variables count
================================================================================
"""
def rbf_kernel(X1, X2, length_scales, sigma_f):
    #number of points in X1 and X2
    N1 = X1.shape[0]
    N2 = X2.shape[0]
    #number of input variables
    D = X1.shape[1]
    #create an empty covariance matrix
    K = np.zeros((N1,N2)) #we will then sum it with respect to input variables
    #loop over every point in X1
    for i in range(N1):
        for j in range(N2):
            squared_dist = 0.0
            #loop over every input variable
            for d in range(D):
                difference = X1[i,d]-X2[j,d]
                normalized_diff = difference/length_scales[d]
                squared_dist += normalized_diff**2
            K[i,j] = sigma_f**2*np.exp(-0.5*squared_dist)
    return K

"""
================================================================================
Next is the function to load the trained model
================================================================================
"""
def load_gp_models(file_path): #rb means read binary
    with open(file_path, "rb") as file:
        all_models = pickle.load(file)
    #the pickle file contains:
    #all_models["current"]; all_models["emittance"]; all_models["envelope"]
    return all_models

"""
================================================================================
The inferencing step (the input is must be scaled)
================================================================================
"""
def predict_one_gp(model, x_star_scaled):
    #x_star_scaled is the input (not included in the training data)
    #first, collect the information stored during training
    X_train = model["X_train"]
    length_scales = model["length_scales"]
    sigma_f = model["sigma_f"]
    L = model["L"]
    alpha = model["alpha"]

    #----------------------------------------------------------------------------
    #Step 1: calculate k_star from x_star (the rbf kernel)
    #----------------------------------------------------------------------------
    k_star = rbf_kernel(X_train, x_star_scaled, length_scales, sigma_f)
    #k_star initially has shape (N,1), because x_star_scaled is one dimensional
    #convert it to shape (N,) basically one dimensional array
    k_star = k_star.reshape(-1)

    #----------------------------------------------------------------------------
    #Step 2: Calculate the predictive mean
    #mean = k_star^T K_y^{-1} y
    #during training, we already calculated \alpha = K_y^{-1} y
    #therefore mean = k_star^T alpha
    #----------------------------------------------------------------------------
    mean_scaled = 0.0

    for i in range(X_train.shape[0]):
        mean_scaled += k_star[i]*alpha[i] #dot product

    #----------------------------------------------------------------------------
    #Step 3: Calculate the predictive variance
    #variance = k_star_star - k_star^T K_y^{-1} k_star
    #for the RBF kernel: k(x_star, x_star) = sigma_f^2
    #----------------------------------------------------------------------------
    #solve L v = k_star
    v = np.linalg.solve(L, k_star)
    #calculate v^T v
    v_squared = 0.0

    for i in range(v.shape[0]):
        v_squared += v[i]**2

    variance_scaled = sigma_f**2 - v_squared
    #a very small negative value might apper due to floating point rounding
    if variance_scaled < 0.0:
        variance_scaled = 0.0
    std_scaled = np.sqrt(variance_scaled)

    return mean_scaled, std_scaled


"""
================================================================================
Denormalization step 
================================================================================
"""
def inverse_scale_prediction(mean_scaled, std_scaled, model):
    y_scaler = model["y_scaler"]

    mean_transformed = y_scaler.inverse_transform(np.array([[mean_scaled]]))[0,0]
    std_transformed = std_scaled/abs(y_scaler.scale_[0])
    output_transform = model.get("output_transform", "none")

    if output_transform == "none":
        #ordinary output for current
        prediction_physical = mean_transformed
        std_physical = std_transformed

        lower_95 = prediction_physical - 1.96*std_physical
        upper_95 = prediction_physical + 1.96*std_physical
    elif output_transform == "log":
        #log output for emittance and envelope
        prediction_physical = np.exp(mean_transformed)
        #the physical distribution is lognormal
        std_physical = np.exp(mean_transformed + 0.5*std_transformed**2)*np.sqrt(np.exp(std_transformed**2)-1.0)
        #lognormal intervals are asymmetric
        lower_95 = np.exp(mean_transformed - 1.96*std_transformed)
        upper_95 = np.exp(mean_transformed + 1.96*std_transformed)
    else:
        raise ValueError(f"Unknown output transformation {output_transform}")

    return prediction_physical, std_physical, lower_95, upper_95
    

"""
================================================================================
Execution step 
================================================================================
"""

# Load all trained GP models
all_models = load_gp_models(
    "trained_models/electron_source_gp_retrain2.pkl"
)

model_current = all_models["current"]
model_emittance = all_models["emittance"]
model_envelope = all_models["envelope"]

# Load the validation dataset
X_validation = np.loadtxt(
    "optimization_results/robust/input_robust_verification.csv",
    delimiter=",",
    dtype=float,
    ndmin=2
)

Y_validation = np.loadtxt(
    "dataset/retrain_cst/output_cst_robust.csv",
    delimiter=",",
    dtype=float,
    ndmin=2
)

# Check the validation dataset
if X_validation.shape[0] != Y_validation.shape[0]:
    raise ValueError(
        "X_validation and Y_validation contain "
        "different numbers of samples."
    )

if X_validation.shape[1] != 7:
    raise ValueError(
        f"Expected 7 input variables, but found "
        f"{X_validation.shape[1]}."
    )

if Y_validation.shape[1] != 3:
    raise ValueError(
        f"Expected 3 outputs, but found "
        f"{Y_validation.shape[1]}."
    )

number_of_validation_samples = X_validation.shape[0]

# Scale every validation input simultaneously
X_scaler = model_current["X_scaler"]

X_validation_scaled = X_scaler.transform(
    X_validation
)

# Create containers
current_pred_cont = np.zeros(number_of_validation_samples)
current_actual_cont = np.zeros(number_of_validation_samples)
current_std_cont = np.zeros(number_of_validation_samples)

emittance_pred_cont = np.zeros(number_of_validation_samples)
emittance_actual_cont = np.zeros(number_of_validation_samples)
emittance_std_cont = np.zeros(number_of_validation_samples)

envelope_pred_cont = np.zeros(number_of_validation_samples)
envelope_actual_cont = np.zeros(number_of_validation_samples)
envelope_std_cont = np.zeros(number_of_validation_samples)

current_lower_95_cont = np.zeros(number_of_validation_samples)
current_upper_95_cont = np.zeros(number_of_validation_samples)

emittance_lower_95_cont = np.zeros(number_of_validation_samples)
emittance_upper_95_cont = np.zeros(number_of_validation_samples)

envelope_lower_95_cont = np.zeros(number_of_validation_samples)
envelope_upper_95_cont = np.zeros(number_of_validation_samples)

for sample_index in range(number_of_validation_samples):

    x_star_scaled = X_validation_scaled[
        sample_index, :
    ].reshape(1, -1)

    y_actual = Y_validation[
        sample_index, :
    ]

    # Predict current
    current_mean_scaled, current_std_scaled = (
        predict_one_gp(
            model_current,
            x_star_scaled
        )
    )

    current_mean, current_std, current_lower_95, current_upper_95 = (
        inverse_scale_prediction(
            current_mean_scaled,
            current_std_scaled,
            model_current
        )
    )

    # Predict emittance
    emittance_mean_scaled, emittance_std_scaled = (
        predict_one_gp(
            model_emittance,
            x_star_scaled
        )
    )

    emittance_mean, emittance_std, emittance_lower_95, emittance_upper_95 = (
        inverse_scale_prediction(
            emittance_mean_scaled,
            emittance_std_scaled,
            model_emittance
        )
    )

    # Predict envelope
    envelope_mean_scaled, envelope_std_scaled = (
        predict_one_gp(
            model_envelope,
            x_star_scaled
        )
    )

    envelope_mean, envelope_std, envelope_lower_95, envelope_upper_95 = (
        inverse_scale_prediction(
            envelope_mean_scaled,
            envelope_std_scaled,
            model_envelope
        )
    )

    # Store results
    current_pred_cont[sample_index] = current_mean
    current_actual_cont[sample_index] = y_actual[0]
    current_std_cont[sample_index] = current_std

    emittance_pred_cont[sample_index] = emittance_mean
    emittance_actual_cont[sample_index] = y_actual[1]
    emittance_std_cont[sample_index] = emittance_std

    envelope_pred_cont[sample_index] = envelope_mean
    envelope_actual_cont[sample_index] = y_actual[2]
    envelope_std_cont[sample_index] = envelope_std

    current_lower_95_cont[sample_index] = (current_lower_95)
    current_upper_95_cont[sample_index] = (current_upper_95)

    emittance_lower_95_cont[sample_index] = (emittance_lower_95)
    emittance_upper_95_cont[sample_index] = (emittance_upper_95)

    envelope_lower_95_cont[sample_index] = (envelope_lower_95)
    envelope_upper_95_cont[sample_index] = (envelope_upper_95)

    print(
        f"Validation sample: {sample_index + 1} "
        "-----------------------"
    )

    print(
        f"Current: mean = {current_mean}, "
        f"actual = {y_actual[0]}, "
        f"standard deviation = {current_std}"
    )

    print(
        f"Emittance: mean = {emittance_mean}, "
        f"actual = {y_actual[1]}, "
        f"standard deviation = {emittance_std}"
    )

    print(
        f"Envelope: mean = {envelope_mean}, "
        f"actual = {y_actual[2]}, "
        f"standard deviation = {envelope_std}"
    )

# Arrange results as 40 rows × 9 columns
stacked_output = np.column_stack([
    current_pred_cont,
    current_actual_cont,
    current_std_cont,
    current_lower_95_cont,
    current_upper_95_cont,

    emittance_pred_cont,
    emittance_actual_cont,
    emittance_std_cont,
    emittance_lower_95_cont,
    emittance_upper_95_cont,

    envelope_pred_cont,
    envelope_actual_cont,
    envelope_std_cont,
    envelope_lower_95_cont,
    envelope_upper_95_cont
])

header = (
    "current_predicted,current_actual,current_std,"
    "current_lower_95,current_upper_95,"
    "emittance_predicted,emittance_actual,emittance_std,"
    "emittance_lower_95,emittance_upper_95,"
    "envelope_predicted,envelope_actual,envelope_std,"
    "envelope_lower_95,envelope_upper_95"
)

np.savetxt(
    "robust_result.csv",
    stacked_output,
    delimiter=",",
    header=header,
    comments=""
)