from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import data_generation as data
import time

max_iter = 10000
X, y = data.generate_time_series_data()
train_X, train_y, valid_X, valid_y, test_X, test_y = data.split_data_for_train_valid_test(X, y)

def train_mlp(train_X, train_y, valid_X, valid_y, alpha, learning_rate, hidden_layers=(7, 5), early_stopping=True, max_iter=10000, random_state=32):
    """
    Trains an MLPRegressor with the given hyperparameters and returns the train and validation MSEs, and the trained model.
    
    Args:
        alpha (float): L2 regularization parameter.
        learning_rate (float): Initial learning rate.
        hidden_layers (tuple): Number of nodes in each hidden layer.
        early_stopping (bool): Whether to use early stopping.
    
    Returns:
        train_mse (float): Mean squared error on the training set.
        valid_mse (float): Mean squared error on the validation set.
        mlp_reg (MLPRegressor): The trained MLPRegressor model.
    """
    mlp_reg = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='logistic',
            alpha=alpha,
            early_stopping=early_stopping,
            max_iter=max_iter,
            random_state=random_state,
            learning_rate_init=learning_rate
        )
    mlp_reg.fit(train_X, train_y)
    train_pred = mlp_reg.predict(train_X)
    train_mse = mean_squared_error(train_y, train_pred)
    valid_pred = mlp_reg.predict(valid_X)
    valid_mse = mean_squared_error(valid_y, valid_pred)
    return train_mse, valid_mse, mlp_reg
     
# choosing the best alpha value
alphas = [0.0001, 0.001, 0.01, 0.04, 0.1]
def alpha_choice(alphas=alphas, learning_rate =0.1, hidden_layers=(7, 7, 7),  early_stopping=True):
    """
    Evaluates different alpha values and prints the generalization gap for each.
    
    Args:
        alphas (list): List of alpha values to test.
        learning_rate (float): Learning rate to use.
        hidden_layers (tuple): Number of nodes in each hidden layer.
        early_stopping (bool): Whether to use early stopping.
    
    Returns:
        train_dict (dict): Mapping of alpha to train MSE.
        valid_dict (dict): Mapping of alpha to validation MSE.
    """
    train_dict = {}
    valid_dict = {}
    for alpha in alphas:
        train_mse, valid_mse,_ = train_mlp(alpha, learning_rate, hidden_layers=hidden_layers, early_stopping=early_stopping)
        train_dict[alpha] = train_mse
        valid_dict[alpha] = valid_mse
        print(f"Alpha: {alpha}, Genralization gap: {valid_mse-train_mse}")
    return train_dict, valid_dict
# train_dict, valid_dict = alpha_choice()

# Choosed alpha = 0.01 based on the genralization gap and to avoid underfitting 

# Choosing the best learning rate 0.1
learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.2]
def lr_convergence_times(alpha=0.01, learning_rates=learning_rates, hidden_layers=(7, 7, 7), early_stopping=True):
    """
    Measures the number of iterations and time to converge for each learning rate.
    
    Args:
        alpha (float): L2 regularization parameter.
        learning_rates (list): List of learning rates to test.
        hidden_layers (tuple): Number of nodes in each hidden layer.
        early_stopping (bool): Whether to use early stopping.
    
    """
    convergence_iters = {}
    convergence_times = {}
    for lr in learning_rates:
        start_time = time.time()
        train_mse, valid_mse, mlp_reg= train_mlp(alpha, lr, hidden_layers=hidden_layers, early_stopping=early_stopping)
        end_time = time.time()
        convergence_iters[lr] = mlp_reg.n_iter_
        convergence_times[lr] = end_time - start_time
        print(f"Learning rate: {lr}, Iterations to converge: {mlp_reg.n_iter_}, Time: {end_time - start_time:.2f} seconds, genralization gap: {valid_mse-train_mse}")




