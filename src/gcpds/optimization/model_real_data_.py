from scipy.sparse import coo_matrix, eye, hstack
from tensorflow.keras.callbacks import Callback
try:
    from google.colab import drive  # only available on Colab
except ImportError:
    drive = None
from tensorflow.python.ops.gen_math_ops import imag_eager_fallback
from tensorflow.python.ops.nn_ops import softmax
from keras import backend as k
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from IPython import display
from scipy.optimize import linprog
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import matplotlib. pyplot as plt
from keras.layers import Activation, Dense, Input, BatchNormalization, Dropout, Conv1D, Flatten, MaxPool1D, Dot, Reshape, Conv2D, Concatenate, ReLU, Lambda, MaxPooling2D, Normalization
import cvxpy as cp
import tensorflow as tf
from keras import Model
from IPython import display
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import warnings
import pandas as pd
from IPython import display
from keras.models import Sequential
import seaborn as sns
import os
from pathlib import Path
import scipy.io as sio
import time

import cvxpy as cp
import subprocess
import sys
import time
import subprocess
import sys
mse = mean_squared_error
mae = mean_absolute_error
mape = mean_absolute_percentage_error
warnings.filterwarnings("ignore")

# Data and output locations, resolved the same way locally, on Colab and on Kaggle.
# RPINN_DATA_DIR overrides the data folder; by default it is RPINN/Data in a clone
# of this repository. RPINN_OUTPUT_DIR is where figures are written (default: cwd).
def _default_data_dir() -> Path:
    env = os.environ.get("RPINN_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "RPINN" / "Data"


DATA_DIR = _default_data_dir()
OUTPUT_DIR = Path(os.environ.get("RPINN_OUTPUT_DIR", "."))
# Definir las librerías que necesitas instalar
libraries = [
    "cvxpy[CLARABEL,CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS]"
]

# Función para instalar las librerías
def install_libraries(libraries):
    for library in libraries:
        subprocess.check_call([sys.executable, "-m", "pip", "install", library])

# Dependencies (including tensorflow==2.15.0) are declared in setup.py; nothing is installed at import time.

def plot_time_p():
    a=plot_time(s=1,tip='M',plotb=False)
    b=plot_time('predict',s=1,plotb=False)
    a =a.rename(columns={'IPOPT(8)': 'IPOPT', 'RPINN(8)': 'RPINN'})
    b =b.rename(columns={'IPOPT(8)': 'IPOPT', 'RPINN(8)': 'RPINN'})
    colors = ['skyblue' if i % 2 == 0 else 'lightgreen' for i in range(len(a.columns))]
    fig, axs = plt.subplots(1, 2,figsize=(22, 10))
    sns.violinplot(a[['IPOPT','RPINN']], inner='box', cut=0, palette=colors, ax=axs[0])
    axs[0].set_yscale('log')
    # Configurar las etiquetas de los ejes y su tamaño de fuente
    # Ajusta aquí el tamaño para los tick labels del eje X

    sns.violinplot(b[['IPOPT','RPINN']], inner='box', cut=0, palette=colors, ax=axs[1])
    axs[1].set_yscale('log')
    # Configurar las etiquetas de los ejes y su tamaño de fuente
    # Ajusta aquí el tamaño para los tick labels del eje X
    axs[1].set_ylim([1e-3,2e1])
    axs[0].set_ylim([1e-3,2e1])
    axs[0].tick_params(axis='x', labelsize=20,rotation=90)
    axs[0].tick_params(axis='y', labelsize=20)

    axs[1].tick_params(axis='x', labelsize=20,rotation=90)
    axs[1].tick_params(axis='y', labelsize=20)
    axs[0].grid(True)
    axs[1].grid(True)
    fig.text(0.5, -0.1, '', ha='center', va='center',fontsize=20)  # Etiqueta común para el eje x centrada
    fig.text(0.05, 0.5, 'Time[s]', ha='center', va='center', rotation='vertical',fontsize=20)  # Etiqueta común para el eje y centrada y rotada

    plt.savefig(f'{OUTPUT_DIR}/Time.pdf',bbox_inches='tight')
    plt.show()
def RGAS():
    Balance1,Weymouth1,PjPi1,Costos1=ng_case_evaluate(s=1,tip1='S',tip2='MAPE',plotb=False)
    Balance2,Weymouth2,PjPi2,Costos2=ng_case_evaluate(s=1,tip1='M',tip2='MAPE',plotb=False)
    Balance3,Weymouth3,PjPi3,Costos3=ng_case_evaluate(s=1,tip1='D',tip2='MAPE',plotb=False)
    Balance1 = Balance1[['IPOPT(8)', 'RPINN(8)']]
    Balance2 = Balance2[['IPOPT(8)', 'RPINN(8)']]
    Balance3 = Balance3[['IPOPT(8)', 'RPINN(8)']]
    Balance3 = Balance3.rename(columns={'IPOPT(8)': 'IPOPT(8) ', 'RPINN(8)': 'RPINN(8) '})
    Balance2 = Balance2.rename(columns={'IPOPT(8)': ' IPOPT(8)', 'RPINN(8)': ' RPINN(8)'})
    Balance1 = Balance1.rename(columns={'IPOPT(8)': '  IPOPT(8) ', 'RPINN(8)': '  RPINN(8) '})
    Weymouth1 = Weymouth1[['IPOPT(8)', 'RPINN(8)']]
    Weymouth2 = Weymouth2[['IPOPT(8)', 'RPINN(8)']]
    Weymouth3 = Weymouth3[['IPOPT(8)', 'RPINN(8)']]
    Weymouth3 = Weymouth3.rename(columns={'IPOPT(8)': 'IPOPT(8) ', 'RPINN(8)': 'RPINN(8) '})
    Weymouth2 = Weymouth2.rename(columns={'IPOPT(8)': ' IPOPT(8)', 'RPINN(8)': ' RPINN(8)'})

    Weymouth1 = Weymouth1.rename(columns={'IPOPT(8)': '  IPOPT(8) ', 'RPINN(8)': '  RPINN(8) '})
    PjPi1 = PjPi1[['IPOPT(8)', 'RPINN(8)']]
    PjPi2 = PjPi2[['IPOPT(8)', 'RPINN(8)']]
    PjPi3 = PjPi3[['IPOPT(8)', 'RPINN(8)']]
    PjPi3 = PjPi3.rename(columns={'IPOPT(8)': 'IPOPT(8) ', 'RPINN(8)': 'RPINN(8) '})
    PjPi2 = PjPi2.rename(columns={'IPOPT(8)': ' IPOPT(8)', 'RPINN(8)': ' RPINN(8)'})
    PjPi1 = PjPi1.rename(columns={'IPOPT(8)': '  IPOPT(8) ', 'RPINN(8)': '  RPINN(8) '})
    Costos1 = Costos1[['IPOPT(8)', 'RPINN(8)']]
    Costos2 = Costos2[['IPOPT(8)', 'RPINN(8)']]
    Costos3 = Costos3[['IPOPT(8)', 'RPINN(8)']]
    Costos3 = Costos3.rename(columns={'IPOPT(8)': 'IPOPT(8) ', 'RPINN(8)': ' RPINN(8) '})
    Costos2 = Costos2.rename(columns={'IPOPT(8)': ' IPOPT(8)', 'RPINN(8)': ' RPINN(8)'})
    Costos1 = Costos1.rename(columns={'IPOPT(8)': '  IPOPT(8) ', 'RPINN(8)': '  RPINN(8)'} )
    Balance = pd.concat([Balance1, Balance2,Balance3], axis=1)
    Weymouth = pd.concat([Weymouth1,Weymouth2, Weymouth3], axis=1)
    PjPi = pd.concat([PjPi1, PjPi2,PjPi3], axis=1)
    Costos = pd.concat([Costos1,Costos2, Costos3], axis=1)
    plots(Balance,Weymouth,PjPi,Costos,s=1,name='RGAS8')

def gen_w(Data_inf, Data_W, Data_T, Data_C, Data_Dc, Data_sto):
    """
    Generates a matrix combining information from wells, pipes, compressors, users, and storages.

    Parameters
    ----------
    Data_inf : List
        Information data.
    Data_W : pd.DataFrame
        Data related to wells.
    Data_T : pd.DataFrame
        Data related to pipes.
    Data_C : pd.DataFrame
        Data related to compressors.
    Data_Dc : pd.DataFrame
        Data related to users.
    Data_sto : pd.DataFrame
        Data related to storages.

    Returns
    -------
    np.ndarray
        A matrix combining all the information.

    Raises
    ------
    ValueError
        If any input data is not in the expected format or type.

    """
    if not all(isinstance(data, pd.DataFrame) for data in [Data_W, Data_T, Data_C, Data_Dc, Data_sto]):
        raise ValueError("All data inputs must be pandas DataFrames")

    num_nodes = len(Data_inf)
    num_wells = len(Data_W)
    wells_matrix = coo_matrix((np.ones(
        num_wells, ), (Data_W['node'] - 1, np.arange(num_wells))), shape=(num_nodes, num_wells))

    num_pipes = len(Data_T)
    pipe_data = np.concatenate(
        (-1.0 * np.ones(num_pipes), np.ones(num_pipes)))
    pipe_row = pd.concat((Data_T['fnode'] - 1, Data_T['tnode'] - 1))
    pipe_col = np.concatenate(2 * [np.arange(num_pipes)])

    pipes_matrix = coo_matrix(
        (pipe_data, (pipe_row, pipe_col)), shape=(num_nodes, num_pipes))

    num_compressors = len(Data_C)
    compressor_data = np.concatenate(
        (-1.0 * np.ones(num_compressors), np.ones(num_compressors)))
    compressor_row = pd.concat((Data_C['fnode'] - 1, Data_C['tnode'] - 1))
    compressor_col = np.concatenate(2 * [np.arange(num_compressors)])
    compressors_matrix = coo_matrix(
        (compressor_data, (compressor_row, compressor_col)), shape=(num_nodes, num_compressors))

    num_users = len(Data_Dc.T)
    users_matrix = hstack(num_users * [eye(num_nodes)])

    num_storages = len(Data_sto)
    storage_matrix = coo_matrix((np.ones(num_storages, ), (
        Data_sto['node'] - 1, np.arange(num_storages))), shape=(num_nodes, num_storages))
    storage_matrix = hstack([storage_matrix, -1.0 * storage_matrix])

    result_matrix = hstack((wells_matrix, pipes_matrix,
                           compressors_matrix, users_matrix, storage_matrix)).toarray()

    return result_matrix


class CustomSigmoidActivation():

    def __init__(self, lower_bound: float, upper_bound: float):
        """
        Initializes the CustomSigmoidActivation with specified bounds.

        Parameters
        ----------
        lower_bound : float
            The lower limit for the activation function.
        upper_bound : float
            The upper limit for the activation function.
        """
        self.lower_bound = tf.cast(lower_bound, tf.float32)
        self.upper_bound = tf.cast(upper_bound, tf.float32)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply the custom sigmoid activation function to the input.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor to apply the activation function.

        Returns
        -------
        tf.Tensor
            The output tensor after applying the activation function.
        """
        x = tf.cast(x, tf.float32)
        x = tf.sigmoid(x) * (self.upper_bound -
                            self.lower_bound) + self.lower_bound
        return x

    def get_config(self) -> dict:
        """
        Returns the configuration of the CustomSigmoidActivation function.

        Returns
                -------
        dict
            A dictionary containing the configuration: lower and upper bounds.
        """
        return {
            "lower_bound": self.lower_bound.numpy(),
            "upper_bound": self.upper_bound.numpy(),
        }


def split(feature_data, output_data):
    """
    Splits the feature and output data into training and test sets.

    Parameters
    ----------
    feature_data : array-like
        The input features data.
    output_data : array-like
        The output data corresponding to the input features.

    Returns
    -------
    tuple
        A tuple containing split data: (X_train, X_test, y_train, y_test)

    """

    X_train, X_test, y_train, y_test = train_test_split(
        feature_data, output_data, test_size=0.2)

    return X_train, X_test, y_train, y_test


def create_customized_dense_layer(input_tensor, layer_info, layer_name, activation_function, initializer):
    """
    Creates a custom dense layer using the Keras Dense function.

    Parameters
    ----------
    input_tensor : Tensor
        The input tensor for the dense layer.
    layer_info : list
        A list containing specific information for constructing the dense layer.
    layer_name : str
        The name of the dense layer.
    activation_function : callable
        The activation function to be applied to the output of the dense layer.
    initializer : Initializer
        The initializer for the kernel weights of the dense layer.

    Returns
    -------
    Tensor
        The output tensor from the dense layer.
    """
    num_neurons = len(layer_info)
    dense_layer = BatchNormalization()(input_tensor)
    dense_layer = Dense(num_neurons, name=layer_name, kernel_initializer=initializer,
                        activation=activation_function)(dense_layer)

    return dense_layer


def create_custom_pres_layer(input_tensor, lower_bound, upper_bound, layer_name, initializer):
    """
    Creates a custom dense layer with a CustomSigmoidActivation activation function.

    Parameters
    ----------
    input_tensor : Tensor
        The input tensor for the dense layer.
    lower_bound : float
        The lower bound for the CustomSigmoidActivation function.
    upper_bound : float
        The upper bound for the CustomSigmoidActivation function.
    layer_name : str
        The name of the dense layer.
    initializer : Initializer
        The initializer for the kernel weights of the dense layer.

    Returns
    -------
    Tensor
        The output tensor from the dense layer.
    """
    num_neurons = len(lower_bound)
    custom_activation = CustomSigmoidActivation(lower_bound, upper_bound)
    dense_layer = Dense(num_neurons, name=layer_name, kernel_initializer=initializer,
                        activation=custom_activation)(input_tensor)

    return dense_layer


def extract_flows_from_model(model, test_data):
    """
    Extracts the output of a specific layer ('Flujos') from a given Keras model.

    Parameters
    ----------
    model : keras.Model
        The Keras model from which to extract the layer output.
    test_data : array-like
        The input data to pass through the model.

    Returns
    -------
    numpy.ndarray
        The output of the 'Flujos' layer in the model.
    """
    intermediate_model = tf.keras.Model(
        inputs=model.inputs, outputs=model.get_layer('Flujos').output)
    layer_output = intermediate_model.predict(test_data, verbose=False)
    return layer_output


class LearningRateReducer(Callback):
    """
    A custom callback to reduce the learning rate based on epoch thresholds.

    Attributes
    ----------
    epoch_threshold : list of int
        Epoch numbers at which the learning rate should be reduced.

    Methods
    -------
    on_epoch_begin(epoch, logs=None):
        Reduces the learning rate if the current epoch is in the epoch_threshold.
    """

    def __init__(self, epoch_threshold):
        """
        Initializes the LRReducer callback with specified epoch thresholds.

        Parameters
        ----------
        epoch_threshold : list of int
            Epoch numbers at which the learning rate should be reduced.
        """
        super(LearningRateReducer, self).__init__()
        self.epoch_threshold = epoch_threshold

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of an epoch during training.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        logs : dict, optional
            Currently no logs are used in this method.
        """
        if epoch in self.epoch_threshold:
            lr = float(self.model.optimizer.lr)
            new_lr = lr * 0.1
            self.model.optimizer.lr.assign(new_lr)


class WellDense(tf.keras.layers.Layer):
    """
    A custom Keras layer that applies a threshold to the sum of inputs.

    Attributes
    ----------
    well : tf.Variable
        The threshold value for the sum of inputs.

    Methods
    -------
    call(inputs):
        Computes the sum of inputs and applies the threshold.
    get_config():
        Returns the layer's configuration.
    """

    def __init__(self, well=80.0, **kwargs):
        """
        Initializes the WellDense layer with a specified threshold.

        Parameters
        ----------
        well : float, optional
            The threshold value for the sum of inputs.
        """
        super().__init__(**kwargs)
        self.well = tf.Variable(
            initial_value=well, trainable=False, dtype=tf.float32)

    def call(self, inputs):
        """
        Computes the sum of inputs and applies the threshold.

        Parameters
        ----------
        inputs : Tensor
            The input tensor to the layer.

        Returns
        -------
        Tensor
            The output tensor after applying the threshold.
        """
        x = inputs
        w = tf.reduce_sum(x, axis=1, keepdims=True)
        w = tf.where(w >= self.well, self.well, w)
        return w

    def get_config(self):
        """
        Returns the configuration of the WellDense layer.

        Returns
        -------
        dict
            A dictionary containing the configuration: well.
        """
        return {
            "well": self.well.numpy(),
        }


class RestrictedDense(tf.keras.layers.Layer):
    """
    A custom dense layer with constraints on its output.

    Attributes
    ----------
    initializer : Initializer
        The initializer for the kernel weights of the layer.
    well : tf.Variable
        A threshold value used in the layer's computations.
    units : int
        The number of units in the dense layer.
    dense : tf.keras.layers.Dense
        The underlying dense layer.

    Methods
    -------
    call(inputs):
        Computes the layer's output with the given inputs.
    get_config():
        Returns the layer's configuration.
    """

    def __init__(self, initializer, well, units, **kwargs):
        """
        Initializes the RestrictedDense layer.

        Parameters
        ----------
        initializer : Initializer
            The initializer for the kernel weights of the layer.
        well : float
            A threshold value used in the layer's computations.
        units : int
            The number of units in the dense layer.
        """
        super().__init__(**kwargs)
        self.initializer = initializer
        self.units = units
        self.well = tf.constant(well, dtype=tf.float32)

    def call(self, inputs):
        """
        Computes the output of the layer with given inputs.

        Parameters
        ----------
        inputs : tuple of Tensors
            Inputs to the layer: (x, In, fw).

        Returns
        -------
        Tensor
            The output tensor of the layer.
        """
        x, In, fw = inputs
        x = tf.where(x <= 0., 0., x)
        x = tf.where(x <= In, x, In)
        mask = tf.where(tf.reduce_sum(In, axis=-1, keepdims=True) <= self.well, 0.0, 1.0)
        x = x * mask
        return x

    def get_config(self):
        """
        Returns the configuration of the RestrictedDense layer.

        Returns
        -------
        dict
            A dictionary containing the configuration.
        """
        return {
            "initializer": self.initializer,
            "well": self.well.numpy(),
            "units": self.units,
        }


def mish(x):
    """
    Mish Activation Function.

    Applies the mish function element-wise to the input Tensor.

    Parameters
    ----------
    x : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The transformed tensor after applying the mish activation function.
    """
    return x * tf.math.tanh(tf.math.softplus(x))


def swish(x):
    """
    Swish Activation Function.

    Applies the swish function element-wise to the input Tensor.

    Parameters
    ----------
    x : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The transformed tensor after applying the swish activation function.
    """
    return x * tf.sigmoid(x)


class CustomLossFuntion(tf.keras.losses.Loss):
    """
    A custom loss function that computes a loss based on specific criteria
    and includes a penalty term for certain conditions.

    Attributes
    ----------
    Kt : float
        A coefficient used in the loss computation.
    i : int
        Index used in gathering elements from predictions.
    j : int
        Another index used in gathering elements from predictions.
    Bc : float
        A boundary condition used in the penalty term.
    ic : int
        Index for penalty computation.
    jc : int
        Another index for penalty computation.
    find : int, optional
        A fixed index to slice the predictions for certain operations.
    """

    def __init__(self, Kt, i, j, Bc, ic, jc, find=6, name="Wey"):
        super().__init__(name=name)
        self.Kt = Kt
        self.i = i
        self.j = j
        self.Bc = Bc
        self.ic = ic
        self.jc = jc
        self.find = find
        self.lower_bound = tf.constant(1.0, dtype=tf.float32)
        self.upper_bound = tf.constant(Bc, dtype=tf.float32)

    def call(self, y_true, y_pred):
        """
        Computes the custom loss between true labels and predicted labels.

        Parameters
        ----------
        y_true : Tensor
            True labels.
        y_pred : Tensor
            Predicted labels.

        Returns
        -------
        Tensor
            Computed loss value.
        """
        f = tf.cast(y_pred[:, :self.find], tf.float32)
        P = tf.cast(y_pred[:, self.find:], tf.float32)
        pi = tf.gather(P, self.i, axis=1)
        pj = tf.gather(P, self.j, axis=1)
        loss = tf.keras.losses.huber(f, tf.sign(pi**2-pj**2)*tf.sqrt(self.Kt*tf.abs(pi**2-pj**2)))

        P_ic = tf.gather(P, self.ic, axis=1)
        P_jc = tf.gather(P, self.jc, axis=1)
        h = self.upper_bound + self.lower_bound
        h1 = self.upper_bound - self.lower_bound
        PjPi = tf.math.divide_no_nan(P_jc, P_ic)  # Asumiendo que P_jc y P_ic están definidos
        PjPi_rounded = tf.round(PjPi * 1000) / 1000
        condition = tf.math.logical_and(  PjPi  >= self.lower_bound,PjPi <= self.upper_bound)
        Penalty = tf.reduce_mean(tf.where(condition, 0.0, tf.square((PjPi-0.5*h)/h1)))
        return 0.3*loss + 0.7*Penalty

    def get_config(self):
        return {
            "Kt": self.Kt,
            "i": self.i,
            "j": self.j,
            "Bc": self.Bc,
            "ic": self.ic,
            "jc": self.jc,
            "find": self.find,
            "s": self.s
        }

class FixedWeightsConstraint(tf.keras.constraints.Constraint):
    def __init__(self, w):
        self.w = tf.constant(w.T, dtype=tf.float32)  # Asegúrate de que `w` está en el formato correcto antes de convertirlo a un tensor.

    def __call__(self, weights):
        return self.w

    def get_config(self):
        return {"w": self.w.numpy().tolist()}  # Convierte el tensor a lista para la serialización.

    @classmethod
    def from_config(cls, config):
        return cls(w=config["w"])

def flow_model(path, fd,a=1e-4,b=1e-4,c=1e-4, s=1):
    """
    Builds and compiles a TensorFlow Keras model based on data from an Excel file and additional parameters.

    Parameters:
    - path (str): Path to the Excel file containing the model's data.
    - fd (array-like): Additional flow data for the model.
    - seeds (int): Seed for random number generation for reproducibility.
    - s (int): Mode switch for the model configuration.

    Returns:
    - tensorflow.keras.Model: The compiled Keras model.
    """
    Data_inf = pd.read_excel(path, sheet_name='node.info')
    Data_D = pd.read_excel(path, sheet_name='node.dem')
    Data_Dc = pd.read_excel(path, sheet_name='node.demcost')
    Data_W = pd.read_excel(path, sheet_name='well')
    Data_T = pd.read_excel(path, sheet_name='pipe')
    Data_C = pd.read_excel(path, sheet_name='comp')
    Data_sto = pd.read_excel(path, sheet_name='sto')
    w = gen_w(Data_inf, Data_W, Data_T, Data_C, Data_Dc, Data_sto)
    N = w.shape[0]
    initializer = tf.keras.initializers.GlorotNormal()
    In = Input(shape=(N,))
    N1 = N
    x=Dense(w.shape[1]*4,'selu',kernel_regularizer=regularizers.L1L2(a,a))(In)
    F = []
    if s == 0:
        Li = np.concatenate((Data_T['Fg_min'].values, [0] * len(Data_C), Data_D['Res'].values * 0.0, Data_D['Ind'].values * 0.0, Data_D['Com'].values * 0.0,
                            Data_D['NGV'].values * 0.0, Data_D['Ref'].values*0.0,
                            Data_D['Pet'].values * 0.0, [0] * len(Data_sto.values) * 2))
        Ls = np.concatenate((Data_T['Fg_max'].values, Data_C['fmaxc'].values, fd, Data_D['Ind'].values * 0.0, Data_D['Com'].values * 0.0, Data_D['NGV'].values * 0.0, Data_D['Ref'].values * 0.0, Data_D['Pet'].values * 0.0,
                            (Data_sto['V0'].values - Data_sto['V_max'].values) * 0.0, (Data_sto['Vmax'].values - Data_sto['V0'].values) * 0.0))
        F = []
        F0 = WellDense(well=Data_W['Imax'].values[0], name='F_0')(In)
        F1 = create_customized_dense_layer(x,Li,'F_1',CustomSigmoidActivation(Li,Ls),initializer)
        F = Concatenate(name='F')([F0, F1])  # ,F2,F3])

    elif s == 1:
        fdin = len(Data_W['Imin'].values) + \
            len([0] * len(Data_T)) + len([0] * len(Data_C))
        fdfn = len([0] * len(Data_sto.values) * 2) + \
            len(Data_inf['Pmin'].values)
        F0 = WellDense(well=Data_W['Imax'].values[0], name='F_0')(In)
        Li = Data_T['Fg_min'].values
        Ls = Data_T['Fg_max'].values
        F1 = create_customized_dense_layer(x,Li,'F_1',CustomSigmoidActivation(Li,Ls),initializer)

        Li = [0] * len(Data_C)
        Ls = Data_C['fmaxc'].values
        F2 = create_customized_dense_layer(x,Li,'F_2',CustomSigmoidActivation(Li,Ls),initializer)
        xt=Dense(N1,'selu',kernel_regularizer=regularizers.L1L2(b,b),kernel_initializer=initializer)(x)
        xt=BatchNormalization()(xt)
        F3 = RestrictedDense(
            initializer=initializer, well=Data_W['Imax'].values[0], name='F_3', units=len(Data_inf))([xt, In, F0])
        Li=np.concatenate((Data_D['Ind'].values*0.0,Data_D['Com'].values*0.0,Data_D['NGV'].values*0.0,Data_D['Ref'].values*0.0,
                        Data_D['Pet'].values*0.0,[0]*len(Data_sto.values)*2))
        Ls=np.concatenate((Data_D['Ind'].values*0.0,Data_D['Com'].values*0.0,Data_D['NGV'].values*0.0,Data_D['Ref'].values*0.0,Data_D['Pet'].values*0.0,
                    (Data_sto['V0'].values-Data_sto['V_max'].values)*0.0,(Data_sto['Vmax'].values-Data_sto['V0'].values)*0.0))
        F4 = create_customized_dense_layer(In,Li,'F_4',CustomSigmoidActivation(Li,Ls),initializer)
        F = Concatenate(name='F')([F0, F1, F2, F3, F4])

    Out1=Dense(units=w.shape[0],use_bias=False,kernel_constraint=FixedWeightsConstraint(w),name='Bal')(F)

    Kt=tf.constant(Data_T['Kij'].values,dtype=tf.float32)
    i=Data_T['fnode'].values-1
    j=Data_T['tnode'].values-1
    ftin=len(Data_W)
    ftfn=len(Data_T)
    f=F[:,ftin:ftfn+ftin]
    Li=Data_inf['Pmin'].values
    Ls=Data_inf['Pmax'].values
    
    
    x1=BatchNormalization()(In)
    x1=Dense(w.shape[1]*4,'selu',kernel_regularizer=regularizers.L1L2(c,c),kernel_initializer=initializer)(x1)
    x1=BatchNormalization()(x1)
    x1=Dense(w.shape[1]*4,'selu',kernel_regularizer=regularizers.L1L2(c,c),kernel_initializer=initializer)(x1)
    x1=BatchNormalization()(x1)
    x1=Dense(w.shape[1]*4,'selu',kernel_regularizer=regularizers.L1L2(c,c),kernel_initializer=initializer)(x1)
    x1=BatchNormalization()(x1)
    
    ib=Data_C['fnode'].values-1
    jb=Data_C['tnode'].values-1
    P =create_custom_pres_layer(x1,Li,Ls,'P',initializer)

    Out2 = Concatenate(name='Wey')([f, P])
    Out3 = P

    Bc = Data_C['ratio'].values

    model = Model(In, [Out1, Out2])
    model.compile(loss={'Bal': 'huber', 'Wey': CustomLossFuntion(Kt,i,j,Bc,ib,jb,ftfn)},
                  optimizer=tf.keras.optimizers.Adamax(1e-2), loss_weights=[0.5, 0.5])
    return model


def plots(Balance, Wey, re, Costos, s,name='RGAS',selec='All'):
    """
    Creates a set of boxplot visualizations for various data frames.

    Parameters:
    - Balance (DataFrame): DataFrame containing balance data.
    - Wey (DataFrame): DataFrame containing Weymouth data.
    - re (DataFrame): DataFrame containing ratio data.
    - Costos (DataFrame): DataFrame containing cost data.
    - s (int): Mode switch for the plot titles.

    This function generates a 2x2 grid of boxplots, each representing data from
    one of the provided DataFrames. The boxplots for 'Balance' and 'Costos' are
    set to a logarithmic scale. The title of the plots changes based on the value of 's'.
    """
    
    if selec == 'All':
        fig, axs = plt.subplots(2, 2, figsize=(19, 19))
        fontsize = 20
        rotation_angle = 90
        colors = ['skyblue' if i % 2 == 0 else 'lightgreen' for i in range(len(Balance.columns))]
        # Datos de ejemplo
        # Asumimos que Balance, Wey, re, y Costos son columnas en un DataFrame llamado df

        # Configurando el violinplot para Balance
        sns.violinplot(data=Balance, inner='box', cut=0, palette=colors, ax=axs[0, 0] )
        axs[0, 0].set_yscale('log')
        axs[0, 0].set_ylabel('MAPE', fontsize=fontsize)
        #axs[0, 0].set_xlabel('Network dimension', fontsize=fontsize)
        axs[0, 0].tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
        axs[0, 0].tick_params(axis='y', labelsize=fontsize)
        axs[0, 0].set_ylim(1e-9, 1e2)
        axs[0, 0].grid(True)
        

        # Configurando el violinplot para Wey
        sns.violinplot(data=Wey, inner='box', cut=0, palette=colors,ax=axs[0, 1])
        axs[0, 1].set_yscale('log')
        axs[0, 1].set_ylabel('MAPE', fontsize=fontsize)
        #axs[0, 1].set_xlabel('Network dimension', fontsize=fontsize)
        axs[0, 1].tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
        axs[0, 1].tick_params(axis='y', labelsize=fontsize)
        axs[0, 1].set_ylim(1e-9, 1e2)
        axs[0, 1].grid(True)
        # Configurando el violinplot para re
        sns.violinplot(data=re, inner='box', cut=0, palette=colors, ax=axs[1, 0])
        axs[1, 0].set_ylabel('MAPE', fontsize=fontsize)
        #axs[1, 0].set_xlabel('Network dimension', fontsize=fontsize)
        axs[1, 0].tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
        axs[1, 0].tick_params(axis='y', labelsize=fontsize)
        axs[1, 0].grid(True)

        # Configurando el violinplot para Costos
        sns.violinplot(data=Costos, inner='box', cut=0, palette=colors, ax=axs[1, 1])
        axs[1, 1].set_yscale('symlog')
        axs[1, 1].set_ylabel('Cost difference', fontsize=fontsize)
        #axs[1, 1].set_xlabel('Network dimension', fontsize=fontsize)
        axs[1, 1].tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
        axs[1, 1].tick_params(axis='y', labelsize=fontsize)
        axs[1, 1].set_ylim(1e4, 1e9)
        axs[1, 1].grid(True)

        # Ajuste de la disposición de la gráfica y establecimiento del título principal basado en la variable 's'
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plt.savefig(f'{OUTPUT_DIR}/'+name+'.pdf',bbox_inches='tight')
        plt.show()
    else:
        fig, axs = plt.subplots(1, 2, figsize=(20,10))
        fontsize = 20
        rotation_angle = 90
        colors = ['skyblue' if i % 2 == 0 else 'lightgreen' for i in range(len(Balance.columns))]
        # Datos de ejemplo
        # Asumimos que Balance, Wey, re, y Costos son columnas en un DataFrame llamado df
        

        # Configurando el violinplot para Wey
        sns.violinplot(data=Wey, inner='box', cut=0, palette=colors,ax=axs[0])
        axs[0].set_yscale('log')
        axs[0].set_ylabel('MAPE', fontsize=fontsize)
        #axs[0, 1].set_xlabel('Network dimension', fontsize=fontsize)
        axs[0].tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
        axs[0].tick_params(axis='y', labelsize=fontsize)
        axs[0].set_ylim(1e-9, 1e2)
        axs[0].grid(True)
        # Configurando el violinplot para Costos
        sns.violinplot(data=Costos, inner='box', cut=0, palette=colors, ax=axs[1])
        axs[1].set_yscale('symlog')
        axs[1].set_ylabel('Cost', fontsize=fontsize)
        #axs[1, 1].set_xlabel('Network dimension', fontsize=fontsize)
        axs[1].tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
        axs[1].tick_params(axis='y', labelsize=fontsize)
        axs[1].set_ylim(1e4, 1e9)
        axs[1].grid(True)
        # Ajuste de la disposición de la gráfica y establecimiento del título principal basado en la variable 's'
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{OUTPUT_DIR}/'+name+'_1'+'.pdf',bbox_inches='tight')
        plt.show()



def load_data(path, key):
    """
    Loads data from a MATLAB file specified by 'path' and extracts data based on 'key'.

    Parameters:
    - path (str): The path to the MATLAB file.
    - key (str): The key corresponding to the data within the MATLAB file.

    Returns:
    - list: A list containing the extracted data.

    The function iterates through eight sets of data within the MATLAB file, extracting
    each set and appending it to a list. The extracted data is expected to be in a
    specific format within the MATLAB file.
    """
    aux = []
    cont = 0
    for i in range(9):
        aux.append([])
        X_test = sio.loadmat(path)[key][0][i]
        for j in X_test:
            aux[cont].append(j)
        cont = cont + 1
    return aux


def evaluate_balance(FT, W, X,s='MAPE'):
    """
    Evaluates the balance of predictions using Mean Absolute Percentage Error (MAPE).

    Parameters:
    - FT (list): A list containing two sets of forecasted data.
    - W (array-like): Weight matrix used in the calculation.
    - X (array-like): Actual data for comparison.

    Returns:
    - list: A list containing two lists of MAPE values for each forecast set in FT.

    This function computes the MAPE between the actual data (X) and each set of
    forecasted data in FT, using the provided weight matrix W.
    """
    if s=='MAPE':
        aux1 = [mape(X[i], FT[0][i] @ W.T) for i in range(len(FT[0]))]
        aux2 = [mape(X[i], FT[1][i] @ W.T) for i in range(len(FT[1]))]
    else:
        aux1 = [mae(X[i], FT[0][i] @ W.T) for i in range(len(FT[0]))]
        aux2 = [mae(X[i], FT[1][i] @ W.T) for i in range(len(FT[1]))]


    return [aux1, aux2]


def evaluate_weymouth(X, Y,s='MAPE'):
    """
    Evaluates the Weymouth equation performance by calculating the MAPE (Mean Absolute Percentage Error)
    between actual values (X) and predicted values (Y).

    Parameters:
    - X (list of lists): Actual values for comparison.
    - Y (list of lists): Predicted values to be compared against X.

    Returns:
    - list of lists: Calculated MAPE values for each set of comparisons.

    The function calculates the MAPE for two sets of data (indicated by indices 0 and 1) in X and Y.
    """
    # Using list comprehensions for more concise code
    if s=='MAPE':
        aux1 = [mape(X[0][i], Y[0][i]) for i in range(len(X[0]))]
        aux2 = [mape(X[1][i], Y[1][i]) for i in range(len(X[1]))]
    else:
        aux1 = [mae(X[0][i], Y[0][i]) for i in range(len(X[0]))]
        aux2 = [mae(X[1][i], Y[1][i]) for i in range(len(X[1]))]

    return [aux1, aux2]


def evaluate_cost(X, Y):
    """
    Calculates the difference between predicted costs and actual costs.

    Parameters:
    - X (list): A list of actual cost values.
    - Y (list): A list of predicted cost values.

    Returns:
    - list: A list of differences between predicted and actual costs.

    This function iterates over each element in X and Y, calculates the
    difference (Y[i] - X[i]) for each corresponding element, and returns
    these differences in a list.
    """
    # Using list comprehension for concise calculation
    return [Y[i] - X[i] for i in range(len(X))]


class TimeHistoryTest(Callback):
    """
    A custom callback to track the time taken for predictions during model inference.

    Attributes:
    - prediction_times (list): A list to store the time taken for each prediction batch.

    Methods:
    - on_predict_begin: Called at the beginning of prediction.
    - on_predict_batch_begin: Called at the beginning of a batch during prediction.
    - on_predict_batch_end: Called at the end of a batch during prediction.
    """

    def on_predict_begin(self, logs=None):
        """
        Initialize the prediction_times list at the beginning of prediction.
        """
        self.prediction_times = []

    def on_predict_batch_begin(self, batch, logs=None):
        """
        Record the start time of a batch prediction.

        Parameters:
        - batch: Index of the batch.
        - logs: Additional information about the batch.
        """
        self.batch_time_start = time.time()

    def on_predict_batch_end(self, batch, logs=None):
        """
        Calculate and record the time taken for a batch prediction.

        Parameters:
        - batch: Index of the batch.
        - logs: Additional information about the batch.
        """
        batch_time_end = time.time() - self.batch_time_start
        self.prediction_times.append(batch_time_end)


def extract_data(times, b):
    """
    Extracts a specific subset of data from a multi-dimensional array.

    Parameters:
    - times (multi-dimensional array): The source array from which data is extracted.
    - b (int): The specific index in the first dimension to extract data from.

    Returns:
    - list: A list containing the extracted data.

    This function iterates over 320 elements in the specified sub-array of 'times',
    extracting a particular value from each element.
    """
    # Using a list comprehension for concise extraction
    return [times[b, :][i][0][0] for i in range(320)]

def plot_time(b='fit', s1=f'{DATA_DIR}/Redes/', s2=f'{DATA_DIR}/', s=1,tip='M',plotb=True):
    """
    Generates and displays boxplot visualizations for time measurements of different network models.

    Parameters:
    - b (str): Indicates the type of operation to perform, 'fit' for fitting or 'predict' for prediction.
    - s1 (str): Base path for the file location.
    - s2 (str): Secondary path for data and model files.
    - s (int): Mode indicator for selecting model paths.

    The function first determines the file paths for the Excel and model files based on the mode 's'.
    It then loads the test data and times data. If 'b' is 'predict', it iterates over models, performs
    predictions, measures the time taken for predictions, and stores these times in a DataFrame.
    This DataFrame is then used to create boxplot visualizations. If 'b' is not 'predict', it loads
    time measurements from a CSV file and creates boxplots for these measurements.

    The boxplots provide a comparison of time taken by different network models for predictions or fitting.
    """

    s1_path = Path(s1)
    s2_path = Path(s2)

    filepath = ['ng_case8.xlsx', 'ng_case9.xlsx', 'ng_case10.xlsx', 'ng_case11.xlsx',
                'ng_case12.xlsx', 'ng_case13.xlsx', 'ng_case14.xlsx', 'ng_case15.xlsx', 'ng_caseCol_18.xlsx']
    if s == 0:
        modelNum='Modelo1'
        modelpath = ['model1_8.h5', 'model1_9.h5', 'model1_10.h5', 'model1_11.h5',
                     'model1_12.h5', 'model1_13.h5', 'model1_14.h5', 'model1_15.h5', 'model1_Col.h5']
    elif s == 1:
        modelNum='Modelo2'
        modelpath = ['model2_8.h5', 'model2_9.h5', 'model2_10.h5', 'model2_11.h5',
                     'model2_12.h5', 'model2_13.h5', 'model2_14.h5', 'model2_15.h5', 'model2_Col.h5']

    X_test = load_data(s2_path /tip/'MAE'/'inputs_P.mat', 'inputs')
    times = np.array(sio.loadmat(s2_path /'Registro'/ 'times_P.mat')['times'])
    fontsize = 20
    rotation_angle = 45
    if b == 'predict':
        Times_p = pd.DataFrame()
        for im in range(9):
            path = s1_path / filepath[im]
            fd = np.array(X_test[im]).max(axis=0)
            model2 = flow_model(str(path), fd, s=s)
            model2.load_weights(s2_path / tip/modelNum/modelpath[im])

            time_callback = TimeHistoryTest()
            predictions = model2.predict(np.array(X_test[im])[0:320], batch_size=1, callbacks=[
                                         time_callback], verbose=False)
            if im == 8:
                Times_p['IPOPT(Col_18)'] = extract_data(times, im)
                Times_p['RPINN(Col_18)'] = time_callback.prediction_times
                
            else:
                Times_p['IPOPT(' + str(im + 8) + ')'] = extract_data(times, im)
                Times_p['RPINN(' + str(im + 8)
                        + ')'] = time_callback.prediction_times
        if plotb:
            plt.figure(figsize=(20, 20))
            colors = ['skyblue' if i % 2 == 0 else 'lightgreen' for i in range(len(Times_p.columns))]
            sns.violinplot(data=Times_p, inner='box', cut=0, palette=colors)
            plt.yscale('log')
            # Ajusta aquí el tamaño para la etiqueta del eje X
            plt.xlabel('Network dimension', fontsize=20)
            # Ajusta aquí el tamaño para la etiqueta del eje Y
            plt.ylabel('Time[s]', fontsize=20)
            # Configurar las etiquetas de los ejes y su tamaño de fuente
            # Ajusta aquí el tamaño para los tick labels del eje X
            plt.xticks(fontsize=20, rotation=45)
            # Ajusta aquí el tamaño para los tick labels del eje Y
            plt.yticks(fontsize=20)
            plt.savefig(f'{OUTPUT_DIR}/TimePredict.pdf',bbox_inches='tight')
            plt.show()
            return Times_p
        else:
            return Times_p
    else:
        N = 15920
        timesr = pd.read_csv(s2_path / 'Registro'/ 'Time.csv').values
        im = pd.DataFrame()
        for i in range(9):
            if i == 8:
                im['RPINN(Col_18)'] = timesr[:, i]
                im['IPOPT(Col_18)'] = extract_data(times, i)
            else:
                im['RPINN(' + str(i + 8) + ')'] = timesr[:, i]
                im['IPOPT(' + str(i + 8) + ')'] = extract_data(times, i)
        if plotb:
            plt.figure(figsize=(20, 20))
            colors = ['skyblue' if i % 2 == 0 else 'lightgreen' for i in range(len(im.columns))]
            sns.violinplot(data=im, inner='box', cut=0, palette=colors)
            plt.yscale('log')
            # Ajusta aquí el tamaño para la etiqueta del eje X
            plt.xlabel('Network dimension', fontsize=20)
            # Ajusta aquí el tamaño para la etiqueta del eje Y
            plt.ylabel('Time[s]', fontsize=20)
            # Configurar las etiquetas de los ejes y su tamaño de fuente
            # Ajusta aquí el tamaño para los tick labels del eje X
            plt.xticks(fontsize=20, rotation=45)
            # Ajusta aquí el tamaño para los tick labels del eje Y
            plt.yticks(fontsize=20)
            plt.savefig(f'{OUTPUT_DIR}/TimeTrain.pdf',bbox_inches='tight')
            plt.show()
            return im
        else:
            return im
def ng_case_evaluate(s1=f'{DATA_DIR}/',tip1='M',tip2='MAPE', s=1,plotb=True,selec='All'):
    """
    Evaluates multiple gas network cases using pre-trained models and generates evaluation metrics.

    Parameters:
    - s1 (str): Base directory path where the network case files are stored.
    - s2 (str): Directory path where input and output data files, and model files are stored.
    - s (int): Mode indicator (0 or 1) to choose between two sets of model paths.

    This function performs the following steps:
    - Loads test input and output data.
    - Based on the mode 's', selects a set of model paths.
    - Iterates over multiple network cases, loading the necessary data for each case.
    - For each case, it loads a pre-trained model and performs predictions.
    - Computes evaluation metrics such as Weymouth, Balance, Cost, and Pressure Ratios (Pj/Pi).
    - Aggregates the results into DataFrames for each metric.
    - Finally, calls the `plots` function to visualize the results.

    Returns a tuple of DataFrames: (Balance, Weymouth, Pj/Pi, Cost)
    """
    s1_path = Path(s1) 
    #s2_path = Path(s1)
    #s3_path = Path(s1)

    X_test = load_data(s1_path / tip1/tip2 /'inputs_P.mat', 'inputs')
    #y_test = load_data(s2_path / 'outputs_P.mat', 'outputs')

    if s == 0:
        modelpath = ['model1_8.h5', 'model1_9.h5', 'model1_10.h5', 'model1_11.h5',
                     'model1_12.h5', 'model1_13.h5', 'model1_14.h5', 'model1_15.h5', 'model1_Col.h5']
    elif s == 1:
        modelpath = ['model2_8.h5', 'model2_9.h5', 'model2_10.h5', 'model2_11.h5',
                     'model2_12.h5', 'model2_13.h5', 'model2_14.h5', 'model2_15.h5', 'model2_Col.h5']
    filepath = ['ng_case8.xlsx', 'ng_case9.xlsx', 'ng_case10.xlsx', 'ng_case11.xlsx',
                'ng_case12.xlsx', 'ng_case13.xlsx', 'ng_case14.xlsx', 'ng_case15.xlsx', 'ng_caseCol_18.xlsx']
    Weymouth = pd.DataFrame()
    Balance = pd.DataFrame()
    Costos = pd.DataFrame()
    PjPi = pd.DataFrame()
        
    Bals=sio.loadmat(s1_path / tip1/tip2 /'Bal_P.mat')['inputs'][0]
    Weys=sio.loadmat(s1_path / tip1/tip2 /'Wey_P.mat')['inputs'][0]
    Coms=sio.loadmat(s1_path / tip1/tip2 /'Com_P.mat')['inputs'][0]
    Co=sio.loadmat(s1_path / tip1/tip2 /'Cost_P.mat')['inputs'][0]
    for im in range(9):
        path = s1_path / 'Redes' / filepath[im]
        Data_inf = pd.read_excel(path, sheet_name='node.info')
        Data_D = pd.read_excel(path, sheet_name='node.dem')
        Data_Dc = pd.read_excel(path, sheet_name='node.demcost')
        Data_W = pd.read_excel(path, sheet_name='well')
        Data_T = pd.read_excel(path, sheet_name='pipe')
        Data_C = pd.read_excel(path, sheet_name='comp')
        Data_sto = pd.read_excel(path, sheet_name='sto')
        Cost = np.concatenate((Data_W['Cg'].values, Data_T['C_O'].values, Data_C['costc'], Data_Dc['al_Res'].values,
                               Data_Dc['al_Ind'].values, Data_Dc['al_Com'].values, Data_Dc[
                                   'al_NGV'].values, Data_Dc['al_Ref'].values, Data_Dc['al_Pet'].values,
                               Data_sto['C_S+'].values - Data_sto['C_V'].values, -1 * (Data_sto['C_S-'] - Data_sto['C_V']).values, Data_sto['C_V'].values)).reshape(-1, 1)
        
        w = gen_w(Data_inf, Data_W, Data_T, Data_C, Data_Dc, Data_sto)
        fd = np.array(X_test[im]).max(axis=0)
        model2 = flow_model(str(path), fd, s=s)
        if s==0:        
            model2.load_weights(s1_path /tip1/'Modelo1'/ modelpath[im])
        else:
            model2.load_weights(s1_path /tip1/'Modelo2'/ modelpath[im])
        model_A = tf.keras.Model(
            inputs=model2.inputs, outputs=model2.get_layer('F').output)
        model_B = tf.keras.Model(
            inputs=model2.inputs, outputs=model2.get_layer('P').output)
        Fe = model_A.predict(np.array(X_test[im]), verbose=False)
        Pe = model_B.predict(np.array(X_test[im]), verbose=False)
        i = Data_T['fnode'].values.astype(int) - 1
        j = Data_T['tnode'].values.astype(int) - 1
        CK = Data_T['Kij'].values.reshape(-1,)
        nod = len(Data_inf)
        Fte = Fe[:, 1:1 + len(Data_T)]
        Pie = Pe[:, i]
        Pje = Pe[:, j]
        Pte = np.sign(Pie**2 - Pje**2) * \
            np.sqrt(CK * np.abs(Pie**2 - Pje**2))

        costos = evaluate_cost(Fe @ Cost[:-1], Co[im].reshape(-1,1))
        ic = Data_C['fnode'].values - 1
        jc = Data_C['tnode'].values - 1
        Bc = Data_C['ratio'].values
        Bc = Data_C['ratio'].values
        are = np.round(Pe[:, jc] / Pe[:, ic], 3)
        ind1 = np.unique(np.concatenate((np.where(~(are[:, 0] <= Bc[0]) | ~(are[:, 0] >= 1.0))[
            0], np.where(~(are[:, 1] <= Bc[1]) | ~(are[:, 1] >= 1.0))[0])))
        re1 = np.zeros(Fe.shape[0])
        if len(ind1) != 0:
            re1[ind1] = 100
        if im <= 7:
            Weymouth['IPOPT(' + str(nod) + ')'] = pd.Series(Weys[im].reshape(-1,))
            Weymouth['RPINN(' + str(nod) + ')'] = pd.Series(evaluate_weymouth_two(Fte, Pte,tip2))
            Balance['IPOPT(' + str(nod) + ')'] =  pd.Series(Bals[im].reshape(-1,))
            Balance['RPINN(' + str(nod) + ')'] = pd.Series(evaluate_balance_two(Fe, w, np.array(X_test[im]),tip2))
            Costos['IPOPT(' + str(nod) + ')'] = pd.Series(Co[im].reshape(-1,))
            Costos['RPINN(' + str(nod) + ')'] = pd.Series(np.array(Fe @ Cost[:-1]).reshape(-1,))
            PjPi['IPOPT(' + str(nod) + ')'] = pd.Series(Coms[im].reshape(-1,))
            PjPi['RPINN(' + str(nod) + ')'] = pd.Series(re1)
            
        else:
            Weymouth['IPOPT(Col_18)'] = pd.Series(Weys[im].reshape(-1,))
            Weymouth['RPINN(Col_18)'] = pd.Series(evaluate_weymouth_two(Fte, Pte,tip2))
            Balance['IPOPT(Col_18)'] =  pd.Series(Bals[im].reshape(-1,))
            Balance['RPINN(Col_18)'] = pd.Series(evaluate_balance_two(Fe, w, np.array(X_test[im]),tip2))
            Costos['IPOPT(Col_18)'] = pd.Series(Co[im].reshape(-1,))
            Costos['RPINN(Col_18)'] = pd.Series(np.array(Fe @ Cost[:-1]).reshape(-1,))     
            PjPi['IPOPT(Col_18)'] = pd.Series(Coms[im].reshape(-1,))
            PjPi['RPINN(Col_18)'] = pd.Series(re1)
            
    if plotb:
        plots(Balance, Weymouth, PjPi, Costos, s=s,selec=selec)
        return(Balance, Weymouth, PjPi, Costos)
    else:
        return(Balance, Weymouth, PjPi, Costos)


def evaluate_balance_two(FT, W, X,s='MAPE'):
    """
    Evaluate the Mean Absolute Percentage Error (MAPE) between actual and predicted data.

    Parameters:
    - FT (list of numpy arrays): List of predicted flow data from models.
    - W (numpy array): Weight matrix used for the calculation.
    - X (list of numpy arrays): List of actual flow data for comparison.

    Returns:
    - list: A list containing MAPE values for each corresponding set of predicted and actual data.

    This function computes the MAPE for each set of predicted data in 'FT' against the actual data in 'X'.
    The computation involves a dot product of predicted data with the transpose of weight matrix 'W'.
    The function iterates over the elements of 'FT' and 'X' to compute MAPE for each set of data.
    """
    # List comprehension for more concise and efficient calculation
    if s=='MAPE':
        return [mape(X[i], FT[i] @ W.T) for i in range(len(FT))]
    else:
        return [mae(X[i], FT[i] @ W.T) for i in range(len(FT))]

def evaluate_weymouth_two(X, Y,s='MAPE'):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between two sets of flow data.

    Parameters:
    - X (list of numpy arrays): The first set of flow data (usually the actual flow data).
    - Y (list of numpy arrays): The second set of flow data (usually the predicted flow data).

    Returns:
    - list: A list of MAPE values, each corresponding to the MAPE between the elements of X and Y.

    The function iterates over the corresponding elements of X and Y, calculating the MAPE
    for each pair of elements. The results are stored and returned in a list.
    """
    # Using list comprehension for efficient computation
    if s=='MAPE':
        return [mape(X[i], Y[i]) for i in range(len(X))]
    else:
        return [mae(X[i], Y[i]) for i in range(len(X))]

def ng_evaluate_atip(s1=f'{DATA_DIR}/',tip1='M',tip2='MAPE', s=1,selec='All'):
    """
    Evaluate different neural network models on various gas network cases and plot their performance metrics.

    Parameters:
    - s1 (str): Base directory path for the model and Excel data files.
    - s2 (str): Path for the input data files.
    - s (int): Scenario indicator for choosing different sets of model paths.

    The function processes multiple gas network cases in a loop. For each case, it loads necessary data,
    constructs and loads a neural network model, and predicts outputs. It then evaluates these predictions
    using custom functions for different metrics such as Weymouth formula, balance, cost, and pressure ratios.
    The function visualizes these metrics using boxplot graphs.

    Returns:
    - tuple: A tuple containing lists of evaluation metrics (Balance, Weymouth, PjPi, Costos) for each network case.
    """
    s1_path = Path(s1) 


    if s == 0:
        modelpath = ['model1_8.h5', 'model1_9.h5', 'model1_10.h5', 'model1_11.h5',
                     'model1_12.h5', 'model1_13.h5', 'model1_14.h5', 'model1_15.h5', 'model1_Col.h5']
    elif s == 1:
        modelpath = ['model2_8.h5', 'model2_9.h5', 'model2_10.h5', 'model2_11.h5',
                     'model2_12.h5', 'model2_13.h5', 'model2_14.h5', 'model2_15.h5', 'model2_Col.h5']
    filepath = ['ng_case8.xlsx', 'ng_case9.xlsx', 'ng_case10.xlsx', 'ng_case11.xlsx',
                'ng_case12.xlsx', 'ng_case13.xlsx', 'ng_case14.xlsx', 'ng_case15.xlsx', 'ng_caseCol_18.xlsx']
    
    Weymouth = []  # pd.DataFrame()
    Balance = []  # pd.DataFrame()
    Costos = []  # pd.DataFrame()
    PjPi = []  # pd.DataFrame()
    X_test_ = sio.loadmat(s1_path / tip1/tip2 /'inputs_None.mat')['inputs']
    for im in range(9):
        path = s1_path / 'Redes' / filepath[im]
        Data_inf = pd.read_excel(path, sheet_name='node.info')
        Data_D = pd.read_excel(path, sheet_name='node.dem')
        Data_Dc = pd.read_excel(path, sheet_name='node.demcost')
        Data_W = pd.read_excel(path, sheet_name='well')
        Data_T = pd.read_excel(path, sheet_name='pipe')
        Data_C = pd.read_excel(path, sheet_name='comp')
        Data_sto = pd.read_excel(path, sheet_name='sto')
        Cost = np.concatenate((Data_W['Cg'].values, Data_T['C_O'].values, Data_C['costc'], Data_Dc['al_Res'].values,
                               Data_Dc['al_Ind'].values, Data_Dc['al_Com'].values, Data_Dc[
                                   'al_NGV'].values, Data_Dc['al_Ref'].values, Data_Dc['al_Pet'].values,
                               Data_sto['C_S+'].values - Data_sto['C_V'].values, -1 * (Data_sto['C_S-'] - Data_sto['C_V']).values, Data_sto['C_V'].values)).reshape(-1, 1)
        w = gen_w(Data_inf, Data_W, Data_T, Data_C, Data_Dc, Data_sto)
        #if im == 8:
        #    X_test = sio.loadmat(
        #        s2_path / f'inputs_None_{im + 8}.mat')['inputs'][0]

        #else:
        #   X_test = sio.loadmat(
        #        s2_path / f'inputs_None_{im + 8}.mat')['inputs']
        X_test=X_test_[0][im]
        fd = np.array(X_test).max(axis=0)
        model2 = flow_model(str(path), fd, s=s)
        if s==0:        
            model2.load_weights(s1_path /tip1/'Modelo1'/ modelpath[im])
        else:
            model2.load_weights(s1_path /tip1/'Modelo2'/ modelpath[im])
        model_A = tf.keras.Model(
            inputs=model2.inputs, outputs=model2.get_layer('F').output)
        model_B = tf.keras.Model(
            inputs=model2.inputs, outputs=model2.get_layer('P').output)
        Fe = model_A.predict(np.array(X_test), verbose=False)
        Pe = model_B.predict(np.array(X_test), verbose=False)
        i = Data_T['fnode'].values.astype(int) - 1
        j = Data_T['tnode'].values.astype(int) - 1
        CK = Data_T['Kij'].values.reshape(-1,)

        Fte = Fe[:, 1:1 + len(Data_T)]
        Pie = Pe[:, i]
        Pje = Pe[:, j]
        Pte = np.sign(Pie**2 - Pje**2) * \
            np.sqrt(CK * np.abs(Pie**2 - Pje**2))       
        balance = evaluate_balance_two(Fe, w, np.array(X_test))
        wey = evaluate_weymouth_two(Fte, Pte)
        ic = Data_C['fnode'].values - 1
        jc = Data_C['tnode'].values - 1
        Bc = Data_C['ratio'].values
        are = np.round(Pe[:, jc] / Pe[:, ic], 3)
        ind1 = np.unique(np.concatenate((np.where(~(are[:, 0] <= Bc[0]) | ~(are[:, 0] >= 1.0))[
                         # .shape[0]/are.shape[0]*100
                         0], np.where(~(are[:, 1] <= Bc[1]) | ~(are[:, 1] >= 1.0))[0])))
        re1 = np.zeros(X_test.shape[0])
        if len(ind1) != 0:
            re1[ind1] = 100
        costos = (Fe @ Cost[:-1]).flatten()
        Weymouth.append(wey)
        Balance.append(balance)
        Costos.append(costos)
        PjPi.append(re1)
    if selec == 'All':
        fig, axs = plt.subplots(2, 2, figsize=(29, 20))
        ax1 = axs[0, 0]
        ax2 = axs[0, 1]
        ax3 = axs[1, 0]
        ax4 = axs[1, 1]

        # ax1.boxplot(Balance)
        colors = ['lightgreen' if i % 2 == 0 else 'lightgreen' for i in range(9)]
        #ax1.boxplot(Balance, boxprops=dict(color='blue'), medianprops=dict(
        #    color='orangered'), whiskerprops=dict(color='blue'))
        sns.violinplot(data=Balance, inner='box', cut=0, palette=colors, ax=ax1)
        fontsize = 20
        rotation_angle = 45
        a = ['RPINN(8)', 'RPINN(9)', 'RPINN(10)', 'RPINN(11)', 'RPINN(12)',
            'RPINN(13)', 'RPINN(14)', 'RPINN(15)', 'RPINN(Col_18)']
        # ax1.set_title('Balance')
        ax1.set_ylabel('MAPE', fontsize=20)
        ax1.set_yscale('log')
        ax1.set_xlabel('Network dimension', fontsize=20)
        ax1.set_xticklabels(a)
        ax1.tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
        ax1.tick_params(axis='y', labelsize=fontsize)

        #ax2.boxplot(Weymouth, boxprops=dict(color='blue'), medianprops=dict(
        #    color='orangered'), whiskerprops=dict(color='blue'))
        sns.violinplot(data=Weymouth, inner='box', cut=0, palette=colors, ax=ax2)
        # ax2.set_title('Weymouth')
        ax2.set_yscale('log')
        ax2.set_ylabel('MAPE', fontsize=20)
        ax2.set_xlabel('Network dimension', fontsize=20)
        ax2.set_xticklabels(a)
        ax2.tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
        ax2.tick_params(axis='y', labelsize=fontsize)

        #ax3.boxplot(PjPi, boxprops=dict(color='blue'), medianprops=dict(
        #    color='orangered'), whiskerprops=dict(color='blue'))
        # ax3.set_title('Pj/Pi')
        sns.violinplot(data=PjPi, inner='box', cut=0, palette=colors, ax=ax3)
        ax3.set_ylabel('MAPE', fontsize=20)
        ax3.set_xlabel('Network dimension', fontsize=20)
        ax3.set_xticklabels(a)
        ax3.tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
        ax3.tick_params(axis='y', labelsize=fontsize)

        #ax4.boxplot(Costos, boxprops=dict(color='blue'), medianprops=dict(
        #    color='orangered'), whiskerprops=dict(color='blue'))
        # ax4.set_title('Costos')#MAPE---COSTOS
        sns.violinplot(data=Costos, inner='box', cut=0, palette=colors, ax=ax4)
        ax4.set_yscale('log')
        ax4.set_ylabel('Costs', fontsize=20)
        ax4.set_xlabel('Network dimension', fontsize=20)
        ax4.set_xticklabels(a)
        ax4.tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
        ax4.tick_params(axis='y', labelsize=fontsize)
        
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)
        ax1.set_ylim(1e-4, 1e2)
        ax2.set_ylim(1e-4, 1e2)

        ax4.set_ylim(1e5, 1e10)
        # Ajustar el diseño para evitar solapamientos
        plt.tight_layout()
        # Ajuste de la disposición de la gráfica y establecimiento del título principal basado en la variable 's'
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plt.savefig(f'{OUTPUT_DIR}/RGASATIP.pdf',bbox_inches='tight')
        plt.show()
        return(Balance, Weymouth, PjPi, Costos)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        ax2 = axs[0]
        ax4 = axs[1]

        # ax1.boxplot(Balance)
        colors = ['lightgreen' if i % 2 == 0 else 'lightgreen' for i in range(9)]
        #ax1.boxplot(Balance, boxprops=dict(color='blue'), medianprops=dict(
        #    color='orangered'), whiskerprops=dict(color='blue'))
        
        fontsize = 20
        rotation_angle = 45
        a = ['RPINN(8)', 'RPINN(9)', 'RPINN(10)', 'RPINN(11)', 'RPINN(12)',
            'RPINN(13)', 'RPINN(14)', 'RPINN(15)', 'RPINN(Col_18)']
        #ax2.boxplot(Weymouth, boxprops=dict(color='blue'), medianprops=dict(
        #    color='orangered'), whiskerprops=dict(color='blue'))
        sns.violinplot(data=Weymouth, inner='box', cut=0, palette=colors, ax=ax2)
        # ax2.set_title('Weymouth')
        ax2.set_yscale('log')
        ax2.set_ylabel('MAPE', fontsize=20)
        #ax2.set_xlabel('Network dimension', fontsize=20)
        ax2.set_xticklabels(a)
        ax2.tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
        ax2.tick_params(axis='y', labelsize=fontsize)

        #ax3.boxplot(PjPi, boxprops=dict(color='blue'), medianprops=dict(
        #    color='orangered'), whiskerprops=dict(color='blue'))
        # ax3.set_title('Pj/Pi')
        

        #ax4.boxplot(Costos, boxprops=dict(color='blue'), medianprops=dict(
        #    color='orangered'), whiskerprops=dict(color='blue'))
        # ax4.set_title('Costos')#MAPE---COSTOS
        sns.violinplot(data=Costos, inner='box', cut=0, palette=colors, ax=ax4)
        ax4.set_yscale('log')
        ax4.set_ylabel('Costs', fontsize=20)
        #ax4.set_xlabel('Network dimension', fontsize=20)
        ax4.set_xticklabels(a)
        ax4.tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
        ax4.tick_params(axis='y', labelsize=fontsize)
        
        
        ax2.grid(True)
        ax4.grid(True)
        ax2.set_ylim(1e-4, 1e2)
        ax4.set_ylim(1e5, 1e10)
        # Ajustar el diseño para evitar solapamientos
        plt.tight_layout()
        # Ajuste de la disposición de la gráfica y establecimiento del título principal basado en la variable 's'
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plt.savefig(f'{OUTPUT_DIR}/RGASATIP_1.pdf',bbox_inches='tight')
        plt.show()
        return(Balance, Weymouth, PjPi, Costos)

def visualize_atipic(dataframes1, dataframes2, columns):
    """
    Identifies and visualizes the percentage of outlier values in specified columns of multiple DataFrames
    in stacked bar plots.

    Parameters:
    - dataframes1 (list of DataFrame): List of first set of DataFrames in which to identify outliers.
    - dataframes2 (list of DataFrame): List of second set of DataFrames in which to identify outliers.
    - columns (list): List of column names in the DataFrames to analyze.

    This function generates stacked bar plots displaying the percentage of outliers in each column
    for both sets of DataFrames.
    """
    # Initialize lists to hold the cumulative outlier percentages for each set of DataFrames
    cumulative_outliers1 = {column: np.zeros(len(dataframes1)) for column in columns}
    cumulative_outliers2 = {column: np.zeros(len(dataframes2)) for column in columns}
    
    # Function to calculate outlier percentages
    def calculate_outliers(dataframe, columns):
        outlier_percentages = []
        for column in columns:
            data_array = dataframe[column].values
            percentage = (data_array[data_array >= 1].shape[0] / data_array.shape[0]) * 100
            outlier_percentages.append(percentage)
        return outlier_percentages
    
    # Calculate outlier percentages for each DataFrame in dataframes1
    for i, dataframe in enumerate(dataframes1):
        outliers = calculate_outliers(dataframe, columns)
        for idx, column in enumerate(columns):
            cumulative_outliers1[column][i] = outliers[idx]
    
    # Calculate outlier percentages for each DataFrame in dataframes2
    for i, dataframe in enumerate(dataframes2):
        outliers = calculate_outliers(dataframe, columns)
        for idx, column in enumerate(columns):
            cumulative_outliers2[column][i] = outliers[idx]
    
    # Create a figure with 1 row and 2 columns of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Function to plot stacked bar plots
    def plot_stacked_bars(ax, cumulative_outliers, title):
        bar_width = 0.35
        bottom = np.zeros(len(columns))
        for i in range(len(dataframes1)):
            percentages = [cumulative_outliers[column][i] for column in columns]
            ax.bar(columns, percentages, bar_width, bottom=bottom)
            bottom += percentages
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(columns, fontsize=10, rotation=45)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True)
    
    # Plot for the first set of DataFrames
    plot_stacked_bars(ax1, cumulative_outliers1, 'Set of DataFrames 1')
    
    # Plot for the second set of DataFrames
    plot_stacked_bars(ax2, cumulative_outliers2, 'Set of DataFrames 2')
    
    # Set common labels
    fig.text(0.5, -0.01, 'Network Dimension', ha='center', fontsize=20)
    fig.text(-0.01, 0.5, 'Percentage Outliers', va='center', rotation='vertical', fontsize=20)
    
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/Atipicos.pdf',bbox_inches='tight')
    plt.show()
def visualize_non_convergence(data_path=f'{DATA_DIR}/', network_range=(0, 9), tips=['D']):
    """
    Visualizes the percentage of non-convergence cases for different network dimensions.

    Parameters:
    - data_path (str): Path to the directory containing input data files.
    - network_range (tuple): Range of network files to analyze (inclusive).
    - tips (list): List of subdirectories representing different types of networks (e.g., ['S', 'M', 'D']).

    The function loads data from specified files and calculates the percentage of
    non-convergence cases for each type in tips. It then generates a bar plot to visually represent this data.
    """
    all_percentages = []
    labels = None

    for tip in tips:
        data = sio.loadmat(data_path + tip + '/MAPE/inputs_None.mat')['inputs']
        percentages = []

        for file_index in range(*network_range):
            non_convergence_percentage = (data[0][file_index].shape[0] / 1000) * 100
            percentages.append(non_convergence_percentage)

            if labels is None:
                label = f'N(Col_18)' if file_index == 16 else f'N({data[0][file_index].shape[1]})'
                labels = [label] if labels is None else labels + [label]
        all_percentages.append(percentages)
        # Etiquetas para cada conjunto de barras
    labels = [f'IPOPT({i+8})' for i in range(len(all_percentages[0]))]
    labels[-1]='IPOPT(Col_18)'
    # Convertir a array de NumPy para facilitar la suma acumulativa
    percentages_array = np.array(all_percentages)

    # Crear el gráfico de barras apiladas
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.35

    # Crear la base para la suma acumulativa
    bottom = np.zeros(len(labels))

    # Dibujar cada fila como una capa en las barras apiladas
    for i, row in enumerate(percentages_array):
        ax.bar(labels, row, bar_width, bottom=bottom, label=f'Row {i+1}')
        bottom += row

    # Añadir etiquetas y título
    ax.set_xlabel('Network Dimension', fontsize=15)
    ax.set_ylabel('Percentage Non-Convergence', fontsize=15)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, rotation=45)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True)

    # Mostrar el gráfico
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/Convergencia.pdf',bbox_inches='tight')
    plt.show()
    # Ejemplo de uso
def dynamic_val(s=1,tip='M',s1=f'{DATA_DIR}/Redes/', s2=f'{DATA_DIR}/'):
    """
    Generates sinusoidal behavior and adjusts random data accordingly.

    Returns:
    - X_test: Randomly generated data.
    """
    # Step 1: Generate desired sinusoidal behavior
    x = np.linspace(0, 2 * np.pi, 1000)  # Generates x values from 0 to 2*pi
    amplitude = 150                         # Sine wave amplitude
    frequency = 1                       # Sine wave frequency
    # Generates the sine wave
    sine_wave = abs(amplitude * np.sin(frequency * x))

    # Step 2: Create a matrix of random data
    data = np.random.rand(1000, 8)  # Random data

    # Step 3: Adjust random data
    # The idea is to scale the data so that its sum along the rows equals the corresponding sine wave value
    # Current sum of data along the rows
    data_sum = np.sum(data, axis=1)
    adjustment_factor = sine_wave / data_sum  # Adjustment factor for each row

    s1_path = Path(s1)
    s2_path = Path(s2)

    # Adjust the data
    for i in range(data.shape[1]):
        data[:, i] *= adjustment_factor
    X_test = data

    modelpath = 'model1_8.h5'
    filepath = 'ng_case8.xlsx'
    path = s1_path / filepath
    Data_inf = pd.read_excel(path, sheet_name='node.info')
    Data_D = pd.read_excel(path, sheet_name='node.dem')
    Data_Dc = pd.read_excel(path, sheet_name='node.demcost')
    Data_W = pd.read_excel(path, sheet_name='well')
    Data_T = pd.read_excel(path, sheet_name='pipe')
    Data_C = pd.read_excel(path, sheet_name='comp')
    Data_sto = pd.read_excel(path, sheet_name='sto')
    Cost = np.concatenate((Data_W['Cg'].values, Data_T['C_O'].values, Data_C['costc'], Data_Dc['al_Res'].values,
                           Data_Dc['al_Ind'].values, Data_Dc['al_Com'].values, Data_Dc[
                               'al_NGV'].values, Data_Dc['al_Ref'].values, Data_Dc['al_Pet'].values,
                           Data_sto['C_S+'].values - Data_sto['C_V'].values, -1 * (Data_sto['C_S-'] - Data_sto['C_V']).values, Data_sto['C_V'].values)).reshape(-1, 1)
    w = gen_w(Data_inf, Data_W, Data_T, Data_C, Data_Dc, Data_sto)
    fd = X_test.max(axis=0)
    # Assuming Fd, Fw, and data are your data.
    plt.figure(figsize=(15, 5))
    model2 = flow_model(path, fd, s=0)
    model2.load_weights(s2_path /tip/'Modelo1'/ modelpath)

    model_A = tf.keras.Model(inputs=model2.inputs,
                             outputs=model2.get_layer('F').output)
    model_B = tf.keras.Model(inputs=model2.inputs,
                             outputs=model2.get_layer('P').output)
    i = len(Data_W) + len(Data_C) + len(Data_T)
    j = len(Data_inf)
    fig, axs = plt.subplots(1, 2,figsize=(20, 8))

    Fd = model_A.predict(X_test, verbose=False)[:, i:i + j]
    axs[0].plot(Fd.sum(axis=1), label="Unsupplied Gas")
    Fw = model_A.predict(X_test, verbose=False)[:, 0]
    axs[0].plot(Fw, label="Dynamic-Gas Supply")
    axs[0].plot(data.sum(axis=1), label="Total Demand")
    axs[1].plot(data.sum(axis=1), label="Total Demand")
    axs[1].plot(Fd.sum(axis=1)+Fw, label="Gas Balance")

    modelpath = 'model2_8.h5'
    model2 = flow_model(path, fd, s=1)
    model2.load_weights(s2_path /tip/'Modelo2'/ modelpath)
    model_A = tf.keras.Model(inputs=model2.inputs,
                             outputs=model2.get_layer('F').output)
    model_B = tf.keras.Model(inputs=model2.inputs,
                             outputs=model2.get_layer('P').output)
    i = len(Data_W) + len(Data_C) + len(Data_T)
    j = len(Data_inf)
    Fd = model_A.predict(X_test, verbose=False)[:, i:i + j]
    axs[0].plot(Fd.sum(axis=1), label="Dynamic-Unsupplied Gas")
    axs[1].plot(Fd.sum(axis=1)+Fw, label="Dynamic Gas Balance")
    axs[0].set_xlabel("Samples", fontsize=20)  # Adds label on x-axis
    axs[0].set_ylabel("Flows", fontsize=20)    # Adds label on y-axis
    axs[0].legend()
    axs[1].set_xlabel("Samples", fontsize=20)  # Adds label on x-axis
    axs[1].set_ylabel("Flows", fontsize=20)    # Adds label on y-axis
    axs[1].legend()           # Displays legend on the plot
    
    plt.savefig(f'{OUTPUT_DIR}/ACTR.pdf',bbox_inches='tight')
    plt.show()
    plt.show()

def loss_val(path=f'{DATA_DIR}/Registro/'):
    """
    Visualizes and compares differences in performance metrics of various models using box plots.

    This function loads performance data from different models from the files Weymouth.xlsx, Balance.xlsx, and Costos.xlsx.
    Then, box plots are generated for each of these metrics, showing how the performance values are distributed among the different models.

    The plots display the Mean Absolute Percentage Error (MAPE) and costs, facilitating visual comparison between the models.
    A logarithmic scale is used to ease the visualization of data that may have a wide range of values.

    The function has no input parameters and does not return any value. The results are directly displayed as plots.
    """

    W = pd.read_csv(
        path+'Weymouth.xlsx')
    W.drop(['Unnamed: 0'], axis=1, inplace=True)
    B = pd.read_csv(
        path+'Balance.xlsx')
    B.drop(['Unnamed: 0'], axis=1, inplace=True)
    C = pd.read_csv(
        path+'Costos.xlsx')
    C.drop(['Unnamed: 0'], axis=1, inplace=True)
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    fontsize = 10
    rotation_angle = 45
    # Organizar los datos en la gráfica y establecer escala logarítmica
    #B.boxplot(ax=axs[0], boxprops=dict(color='blue'), medianprops=dict(
    #    color='orangered'), whiskerprops=dict(color='blue'))
    sns.violinplot(data=B, inner='box', cut=0, color='skyblue', ax=axs[0],width=1.5)
    
    axs[0].set_yscale('log')
    axs[0].set_ylabel('MAPE', fontsize=15)
    axs[0].set_xlabel('Loss Functions', fontsize=15)
    axs[0].tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
    axs[0].tick_params(axis='y', labelsize=fontsize)
    # for label in axs[0, 0].get_xticklabels():
    #    label.set_rotation(rotation_angle)

    #W.boxplot(ax=axs[1], boxprops=dict(color='blue'), medianprops=dict(
    #    color='orangered'), whiskerprops=dict(color='blue'))
    #axs[0, 1].set_title('Weymouth')
    sns.violinplot(data=W, inner='box', cut=0, color='skyblue', ax=axs[1],width=1.5)
    
    axs[1].set_yscale('log')
    axs[1].set_ylabel('MAPE', fontsize=15)
    axs[1].set_xlabel('Loss Functions', fontsize=15)
    axs[1].tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
    axs[1].tick_params(axis='y', labelsize=fontsize)
    # for label in axs[0, 1].get_xticklabels():

    #C.boxplot(ax=axs[2], boxprops=dict(color='blue'), medianprops=dict(
    #    color='orangered'), whiskerprops=dict(color='blue'))
    #axs[1, 1].set_title('Costos')
    sns.violinplot(data=C, inner='box', cut=0, color='skyblue', ax=axs[2],width=1.5)
    
    axs[2].set_yscale('symlog')
    axs[2].set_ylabel('Costs', fontsize=15)
    axs[2].set_xlabel('Loss Functions', fontsize=15)
    axs[2].tick_params(axis='x', labelsize=fontsize, rotation=rotation_angle)
    axs[2].tick_params(axis='y', labelsize=fontsize)
    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    axs[0].set_ylim([0,1e6])
    axs[1].set_ylim([0,1e6])
    axs[2].set_ylim([0,1e9])
    plt.savefig(f'{OUTPUT_DIR}/LOSSM.pdf',bbox_inches='tight')
    plt.show()

def bounded(s1=f'{DATA_DIR}/',tip='D',s=1,im=8):
    """
     Generates box plots to visualize and compare performance metric differences across various models.

     This function loads performance data of different models from the Weymouth.xlsx, Balance.xlsx, and Costos.xlsx files.
     Then, box plots are generated for each of these metrics, showing how the performance values are distributed
     among the different models.

     The plots display the Mean Absolute Percentage Error (MAPE) and costs, facilitating visual comparison between the models.
     A logarithmic scale is used to ease the visualization of data that may have a wide range of values.

     Parameters
     ----------
     s1 : str, optional
         Path to the directory containing input data files. Default is f'{DATA_DIR}/'.
     s2 : str, optional
         Path to the directory containing model data files. Default is f'{DATA_DIR}/Registro/'.
     s : int, optional
         Determines the type of model. If 0, uses one set of models, if 1, uses another set. Default is 1.

     Returns
     -------
     None
         The results are directly displayed as plots.
     """
    s1_path = Path(s1)
    X_test = load_data(s1_path / tip/'MAE' /'inputs_P.mat', 'inputs')

    if s == 0:
        modelpath = ['model1_8.h5', 'model1_9.h5', 'model1_10.h5', 'model1_11.h5',
                     'model1_12.h5', 'model1_13.h5', 'model1_14.h5', 'model1_15.h5', 'model1_Col.h5']
    elif s == 1:
        modelpath = ['model2_8.h5', 'model2_9.h5', 'model2_10.h5', 'model2_11.h5',
                     'model2_12.h5', 'model2_13.h5', 'model2_14.h5', 'model2_15.h5', 'model2_Col.h5']
    filepath = ['ng_case8.xlsx', 'ng_case9.xlsx', 'ng_case10.xlsx', 'ng_case11.xlsx','ng_case12.xlsx', 'ng_case13.xlsx', 'ng_case14.xlsx', 'ng_case15.xlsx', 'ng_caseCol_18.xlsx']
    Weymouth = pd.DataFrame()
    Balance = pd.DataFrame()
    Costos = pd.DataFrame()
    PjPi = pd.DataFrame()
    im = im
    path = s1_path / 'Redes' / filepath[im]
    Data_inf = pd.read_excel(path, sheet_name='node.info')
    Data_D = pd.read_excel(path, sheet_name='node.dem')
    Data_Dc = pd.read_excel(path, sheet_name='node.demcost')
    Data_W = pd.read_excel(path, sheet_name='well')
    Data_T = pd.read_excel(path, sheet_name='pipe')
    Data_C = pd.read_excel(path, sheet_name='comp')
    Data_sto = pd.read_excel(path, sheet_name='sto')
    Cost = np.concatenate((Data_W['Cg'].values, Data_T['C_O'].values, Data_C['costc'], Data_Dc['al_Res'].values,
                           Data_Dc['al_Ind'].values, Data_Dc['al_Com'].values, Data_Dc[
                               'al_NGV'].values, Data_Dc['al_Ref'].values, Data_Dc['al_Pet'].values,
                           Data_sto['C_S+'].values - Data_sto['C_V'].values, -1 * (Data_sto['C_S-'] - Data_sto['C_V']).values, Data_sto['C_V'].values)).reshape(-1, 1)
    w = gen_w(Data_inf, Data_W, Data_T, Data_C, Data_Dc, Data_sto)
    fd = np.array(X_test[im]).max(axis=0)
    model2 = flow_model(str(path), fd, s=s)
    model2.load_weights(s1_path/tip /'Modelo2'/ modelpath[im])
    model_A = tf.keras.Model(inputs=model2.inputs,
                             outputs=model2.get_layer('F').output)
    model_B = tf.keras.Model(inputs=model2.inputs,
                             outputs=model2.get_layer('P').output)
    Fe = model_A.predict(np.array(X_test[im]), verbose=False)
    Pe = model_B.predict(np.array(X_test[im]), verbose=False)
    i = Data_T['fnode'].values.astype(int) - 1
    j = Data_T['tnode'].values.astype(int) - 1
    CK = Data_T['Kij'].values.reshape(-1,)
    nod = len(Data_inf)
    Fte = Fe[:, 1:1 + len(Data_T)]
    Fce = Fe[:, 1 + len(Data_T):1 + len(Data_T) + len(Data_C)]
    Fwe = Fe[:, 0:1]

    fig, axs = plt.subplots(2, 2, figsize=(19, 19))
    fontsize = 20
    rotation_angle = 45
    # Organizar los datos en la gráfica y establecer escala logarítmica
    #axs[0, 0].boxplot(Fwe, boxprops=dict(color='blue'), medianprops=dict(
    #    color='orangered'), whiskerprops=dict(color='blue'))
    sns.violinplot(data=Fwe, inner='box', cut=0, color='lightgreen', ax=axs[0,0])
    axs[0, 0].set_xticklabels(np.arange(1, 2))
    #axs[0, 0].set_yscale('symlog')
    axs[0, 0].plot(0, Data_W['Imax'].values, 'k*', markersize=15)
    axs[0, 0].plot(0, 1, 'k*', markersize=15)
    axs[0, 0].set_ylabel('Gas injection[MMSCFD]', fontsize=20)
    axs[0, 0].set_xlabel('Injection node', fontsize=20)
    axs[0, 0].tick_params(axis='x', labelsize=fontsize)
    axs[0, 0].tick_params(axis='y', labelsize=fontsize)

    # for label in axs[0, 0].get_xticklabels():
    #    label.set_rotation(rotation_angle)

    Li = Data_T['Fg_min'].values
    Ls = Data_T['Fg_max'].values

    #axs[0, 1].boxplot(Fte, boxprops=dict(color='blue'), medianprops=dict(
    #    color='orangered'), whiskerprops=dict(color='blue'))
    sns.violinplot(data=Fte, inner='box', cut=0, color='lightgreen', ax=axs[0,1])
    axs[0, 1].set_xticklabels(np.arange(2, 2 + len(Data_T)))
    #axs[0, 1].set_title('Weymouth')
    #axs[0, 1].set_yscale('symlog')
    axs[0, 1].plot(np.arange(0, len(Data_T)), Ls,
                   'k*', markersize=15)
    axs[0, 1].plot(np.arange(0, len(Data_T)), Li, 'k*', markersize=15)
    axs[0, 1].set_ylabel('Gas flow[MMSCFD]', fontsize=20)
    axs[0, 1].set_xlabel('Pipeline node', fontsize=20)
    axs[0, 1].tick_params(axis='x', labelsize=fontsize)
    axs[0, 1].tick_params(axis='y', labelsize=fontsize)
    # for label in axs[0, 1].get_xticklabels():

    Li = [0] * len(Data_C)
    Ls = Data_C['fmaxc'].values
    #axs[1, 0].boxplot(Fce, boxprops=dict(color='blue'), medianprops=dict(
    #    color='orangered'), whiskerprops=dict(color='blue'))
    sns.violinplot(data=Fce, inner='box', cut=0, color='lightgreen', ax=axs[1,0])
    #axs[1, 1].set_title('Costos')
    axs[1, 0].set_xticklabels(
        np.arange(len(Data_T)+2, 2+len(Data_T) + len(Data_C)))
    #axs[1, 0].set_yscale('symlog')
    axs[1, 0].plot(np.arange(0, len(Data_C)), Ls, 'k*',
                   markersize=15)
    axs[1, 0].plot(np.arange(0, len(Data_C)), Li, 'k*', markersize=15)
    axs[1, 0].set_ylabel('Gas flow[MMSCFD]', fontsize=20)
    axs[1, 0].set_xlabel('Compressor node', fontsize=20)
    axs[1, 0].tick_params(axis='x', labelsize=fontsize)
    axs[1, 0].tick_params(axis='y', labelsize=fontsize)
    Li = Data_inf['Pmin'].values
    Ls = Data_inf['Pmax'].values
    #axs[1, 1].boxplot(Pe, boxprops=dict(color='blue'), medianprops=dict(
    #    color='orangered'), whiskerprops=dict(color='blue'))
    sns.violinplot(data=Pe, inner='box', cut=0, color='lightgreen', ax=axs[1, 1])
    #axs[1, 1].set_title('Costos')
    #axs[1, 1].set_yscale('log')
    axs[1, 1].set_xticklabels(np.arange(1, Pe.shape[1]+1))
    axs[1, 1].plot(np.arange(0, Pe.shape[1]), Ls+10, 'k*',
                   markersize=15)
    #axs[1, 1].plot(np.arange(0, Pe.shape[1]), Li, 'k*',
    #               markersize=15, label='Pressure limit')
    axs[1, 1].set_ylabel('Pressure[psia]', fontsize=20)
    axs[1, 1].set_xlabel('Pressure node', fontsize=20)
    axs[1, 1].tick_params(axis='x', labelsize=fontsize)
    #axs[1, 1].set_yscale('log')
    axs[1, 1].tick_params(axis='y', labelsize=fontsize)
    axs[1, 1].grid(True)
    axs[1, 0].grid(True)
    axs[0, 1].grid(True)
    axs[0, 0].grid(True)
    plt.savefig(f'{OUTPUT_DIR}/LIM.pdf',bbox_inches='tight')
    plt.show()
    return Pe
