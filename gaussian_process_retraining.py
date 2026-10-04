import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import time

waktu0 = time.time()
#similar to gp_training, but no train-test splitting. direct training from the given data.

"""
================================================================================
================================================================================
#so X1 and X2 is a two dimensional matrix
#the first index represent the dataset
#the second index represent the input variables count
================================================================================
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
================================================================================
for calculating the negative log marginal likelihood for one Gaussian process
log params are the logarithm of [length_scale_1, ... , length_scale_D, sigma_f, sigma_n]
================================================================================
================================================================================
"""
def negative_log_marginal_likelihood(log_params, X, y):
    N = X.shape[0]
    D = X.shape[1]

    #recover the actual GP hyperparameters
    length_scales = np.zeros(D) 
    for d in range(D):
        length_scales[d] = np.exp(log_params[d])
    sigma_f = np.exp(log_params[D]) #the variance amplitude of RBF kernel
    sigma_n = np.exp(log_params[D+1]) #the amplitude of the noise

    #----------------------------------------------
    #Step 1: build the kernel matrix
    #----------------------------------------------
    K = rbf_kernel(X,X,length_scales, sigma_f)

    #----------------------------------------------
    #Step 2: add noise variance to the diagonal
    #----------------------------------------------
    for i in range(N):
        K[i,i] = K[i,i] + sigma_n**2
    #for numerical stability, add small number
    for i in range(N):
        K[i,i] = K[i,i] + 1e-10

    #----------------------------------------------
    #Step 3: Cholesky decomposition, we are using numpy
    #----------------------------------------------
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return 1e20

    #----------------------------------------------
    #Step 4: Calculate \alpha = K^{-1}y
    #instead of calculating the inverse of K directly, solve:
    #L z = y
    #then:
    #L^T \alpha = z (Cholesky decomposition trick, since the original matrix is symmetric and positive definite)
    #----------------------------------------------
    z = np.linalg.solve(L,y)
    alpha = np.linalg.solve(L.T, z)

    #----------------------------------------------
    #step 5: Calculate the next terms
    #y^T K^{-1} y. 
    #because \alpha = K^{-1} y, then what we calculate is y^T \alpha.
    #----------------------------------------------
    y_Kinv_y = 0.0
    for i in range(N):
        y_Kinv_y += y[i]*alpha[i] #just a matrix multiplication
    data_fit_term = 0.5*y_Kinv_y

    #----------------------------------------------
    #Step 6: Calculate log(det(K)
    #since K = L L^T (Cholesky decomposition)
    #then det(K) = det(L)^2
    #Additionally, since L is triangular, det(L) = product of diagonal elements
    #Thus, log(det(K)) = 2*sum(log(L_ii))
    #----------------------------------------------
    log_determinant_K = 0.0
    for i in range(N):
        log_determinant_K += 2.0*(np.log(L[i,i]))
    complexity_term = 0.5*log_determinant_K

    #----------------------------------------------
    #Step 7: Constant term
    #N/2 * log(2*pi)
    #----------------------------------------------
    constant_term = 0.5*N*np.log(2*np.pi)

    #----------------------------------------------
    #Thus, we get the negative log marginal likelihood, which we will minimize
    #----------------------------------------------

    negative_log_likelihod = data_fit_term + complexity_term + constant_term

    return negative_log_likelihod

"""
================================================================================
================================================================================
GP TRAINING
Train one Gaussian Process, returns optimized hyperparameters
and quantities needed for prediction
================================================================================
================================================================================
"""
def train_gp(X,y):
    #first, the initial guess.
    #since the input X has been scaled to [0,1]
    #length_scale = 0.2 is a reasonable starting point
    #in addition, since Y has also been scaled to [0,1]
    #sigma_f = 1 is reasonable
    #we also assume a very small initial numerical noise
    
    n_features = X.shape[1] #input dimension
    initial_length_scales = np.full(n_features, 0.2) #initial guess for length scale
    initial_sigma_f = 1.0 #initial guess for variance amplitude
    initial_sigma_n = 1e-4 #initial guess for the noise

    #combine into one (logarithmic)
    initial_params = np.concatenate([
        np.log(initial_length_scales),
        [np.log(initial_sigma_f)],
        [np.log(initial_sigma_n)]
    ])

    #----------------------------------------------
    #hyperparameter and negative log marginal likelihood record during optimization
    #----------------------------------------------
    nll_history = []

    #record the initial NLL before optimization
    initial_nll = negative_log_marginal_likelihood(initial_params, X, y)
    nll_history.append(initial_nll)

    def optimization_callback(current_log_params):
        current_nll = negative_log_marginal_likelihood(current_log_params, X, y)
        nll_history.append(current_nll)
        iteration = len(nll_history) - 1
        print(
            f"Iteration {iteration}: "
            f"NLL = {current_nll:.8f}"
        )

    #----------------------------------------------
    #optimize the marginal likelihood 
    #----------------------------------------------
    result = minimize(
        negative_log_marginal_likelihood,
        initial_params,
        args = (X,y),
        method = "L-BFGS-B",
        callback = optimization_callback,
        options = {"maxiter":1000}
    )
    if not result.success:
        print(
            "Warning: optimizer did not fully converge"
        )
        print(result.message)

    #----------------------------------------------
    #get the optimized hyperparameters
    #----------------------------------------------
    optimized_params = result.x
    length_scales = np.exp(optimized_params[:n_features])
    sigma_f = np.exp(optimized_params[n_features])
    sigma_n = np.exp(optimized_params[n_features+1])

    #----------------------------------------------
    #construct final training covariance matrix, then add the noise
    #----------------------------------------------
    N = X.shape[0]
    K = rbf_kernel(X, X, length_scales, sigma_f)
    for i in range(N):
        K[i,i] = K[i,i] + sigma_n**2 + 1e-10

    #----------------------------------------------
    #Stop 4: Cholesky decomposition
    #----------------------------------------------
    L = np.linalg.cholesky(K)

    #----------------------------------------------
    #Step 5: We want alpha = K^{-1} y, just like before
    #----------------------------------------------
    # L z = y
    z = np.linalg.solve(L, y)
    #L^T \alpha = z
    alpha = np.linalg.solve(L.T,z)

    #----------------------------------------------
    #then, make the output
    #----------------------------------------------
    model = {
        "X_train": X,
        "y_train": y,
        "length_scales": length_scales,
        "sigma_f": sigma_f,
        "sigma_n": sigma_n,
        "L": L,
        "alpha": alpha,
        "log_marginal_likelihood": -result.fun
    }
    return model

"""
================================================================================
================================================================================
Scale the input and output data to [0,1], then train one GP
================================================================================
================================================================================
"""
def train_gp_with_scaling(X,y,output_transform = "none"):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if X.shape[0] != y.shape[0]: #different input-output pairs
        raise ValueError("Input and output dimensions are not identical")

    #Fit and apply the input scaler
    X_scaler = MinMaxScaler(feature_range=(0.0,1.0))
    X_scaled = X_scaler.fit_transform(X)
    
    #we are going to use logarithmic output
    if output_transform == "log":
        #the logarithm requires strictly positive values
        if np.any(y <= 0):
            raise ValueError("the argument should be positive, why negative?")
        y_transformed = np.log(y)
    
    elif output_transform == "none":
        y_transformed = y.copy()
    
    else:
        raise ValueError("output_transform must be either 'none' or 'log'")
    
    #do the same for output data (but the input scaler requires two dimensional array, so we need to retransform)
    y_scaler = MinMaxScaler(feature_range=(0.1,0.9))
    y_scaled = y_scaler.fit_transform(y_transformed.reshape(-1,1)).reshape(-1)

    #train the GP using the scaled dataßß
    model = train_gp(X_scaled, y_scaled)

    #add the scaler information, then save the model
    model["X_scaler"] = X_scaler
    model["y_scaler"] = y_scaler
    model["n_features"] = X.shape[1]
    model["output_transform"] = output_transform

    return model

def load_training_data(input_file, output_file):
    #X = np.loadtxt(input_file, delimiter="\t", skiprows=1, dtype=float, ndmin=2)
    X = np.loadtxt(input_file, delimiter=",", dtype=float, ndmin=2)
    Y = np.loadtxt(output_file, delimiter=",", dtype=float, ndmin=2)

    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of input variables: {X.shape[1]}")
    print(f"Number of output variables: {Y.shape[1]}")

    return X, Y

def save_gp_models(file_path, model_current, model_emittance, model_envelope):
    all_models = {
        "current": model_current,
        "emittance": model_emittance,
        "envelope": model_envelope
    }
    file_path = Path(file_path)
    #create the folder if it does not already exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    #write a binary file "wb"
    with open(file_path, "wb") as file:
        pickle.dump(
            all_models, file, protocol=pickle.HIGHEST_PROTOCOL
        )
    print(f"Models saved to: {file_path}")

def split_data(X_input, Y_output, valid_fraction=0.2, seed=42):
    N = X_input.shape[0] #total number of samples
    n_validation = int(N*valid_fraction) #fraction of validation data
    rng = np.random.default_rng(seed) #random number generation
    indices = np.arange(N) #array [0,...,N-1]
    rng.shuffle(indices) #shuffle the index

    validation_indices = indices[:n_validation] #first n indices are for validation
    train_indices = indices[n_validation:] #the rest are for training

    X_train = X_input[train_indices]
    X_validation = X_input[validation_indices]
    Y_train = Y_output[train_indices]
    Y_validation = Y_output[validation_indices]

    print(f"Total samples: {N}")
    print(f"Training samples: {N-n_validation}")
    print(f"Validation samples: {n_validation}")

    np.savetxt("dataset/X_train.csv", X_train, delimiter =",")
    np.savetxt("dataset/X_validation.csv", X_validation, delimiter =",")
    np.savetxt("dataset/Y_train.csv", Y_train, delimiter =",")
    np.savetxt("dataset/Y_validation.csv", Y_validation, delimiter =",")

    return X_train, X_validation, Y_train, Y_validation


##############################################################
#Let's execute the script
##############################################################

X, Y = load_training_data("dataset/input_retrain1_combined.csv",
                          "dataset/output_retrain1_combined.csv")

#X_train, X_validation, Y_train, Y_validation = split_data(X, Y)

model_current = train_gp_with_scaling(X, Y[:,0], output_transform = "none")
model_emittance = train_gp_with_scaling(X, Y[:,1], output_transform = "log")
model_envelope = train_gp_with_scaling(X, Y[:,2], output_transform = "log")

save_gp_models(
    "trained_models/electron_source_gp_retrain1.pkl",
    model_current, model_emittance, model_envelope
)
waktuf = time.time()

print("Durasi:", waktuf-waktu0)
