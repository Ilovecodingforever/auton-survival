
import torch
import numpy as np

from .binary_torch import BinarySurvivalClassifierTorch
from .utilities import train_binary_survival, predict_survival, bce_loss

from auton_survival.utils import _dataframe_to_array
from auton_survival.models.dsm.utilities import _get_padded_features
from auton_survival.models.dsm.utilities import _get_padded_targets
from auton_survival.models.dsm.utilities import _reshape_tensor_with_nans


class BinarySurvivalClassifier:

  def __init__(self, layers=None, random_seed=0):

    self.layers = layers
    self.fitted = False
    self.random_seed = random_seed

  def __call__(self):
    if self.fitted:
      print("A fitted instance of the BinarySurvivalClassifier")
    else:
      print("An unfitted instance of the BinarySurvivalClassifier")

    print("Hidden Layers:", self.layers)

  def _preprocess_test_data(self, x):
    x = _dataframe_to_array(x)
    return torch.from_numpy(x).float()

  def _preprocess_training_data(self, x, t, e, vsize, val_data, random_seed):

    x = _dataframe_to_array(x)
    t = _dataframe_to_array(t)
    e = _dataframe_to_array(e)

    idx = list(range(x.shape[0]))

    np.random.seed(random_seed)
    np.random.shuffle(idx)

    x_train, t_train, e_train = x[idx], t[idx], e[idx]

    x_train = torch.from_numpy(x_train).float()
    t_train = torch.from_numpy(t_train).float()
    e_train = torch.from_numpy(e_train).float()

    if val_data is None:

      vsize = int(vsize*x_train.shape[0])
      x_val, t_val, e_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:]

      x_train = x_train[:-vsize]
      t_train = t_train[:-vsize]
      e_train = e_train[:-vsize]

    else:

      x_val, t_val, e_val = val_data

      x_val = _dataframe_to_array(x_val)
      t_val = _dataframe_to_array(t_val)
      e_val = _dataframe_to_array(e_val)

      x_val = torch.from_numpy(x_val).float()
      t_val = torch.from_numpy(t_val).float()
      e_val = torch.from_numpy(e_val).float()

    return (x_train, t_train, e_train, x_val, t_val, e_val)

  def _gen_torch_model(self, inputdim, optimizer, n_bins, survival_estimator):
    """Helper function to return a torch model."""
    # Add random seed to get the same results like in dcm __init__.py
    np.random.seed(self.random_seed)
    torch.manual_seed(self.random_seed)

    return BinarySurvivalClassifierTorch(inputdim, layers=self.layers,
                                         optimizer=optimizer,
                                         n_bins=n_bins,
                                         survival_estimator=survival_estimator)

  def fit(self, x, t, e, n_bins=20, survival_estimator='km',
          vsize=0.15, val_data=None, event_horizon_time=10*365.25,
          iters=1, learning_rate=1e-3, batch_size=100,
          optimizer="Adam"):

    processed_data = self._preprocess_training_data(x, t, e,
                                                    vsize, val_data,
                                                    self.random_seed)

    x_train, t_train, e_train, x_val, t_val, e_val = processed_data

    #Todo: Change this somehow. The base design shouldn't depend on child

    inputdim = x_train.shape[-1]

    model = self._gen_torch_model(inputdim, optimizer, n_bins, survival_estimator)

    model, _ = train_binary_survival(model,
                                    (x_train, t_train, e_train),
                                    (x_val, t_val, e_val),
                                    epochs=iters,
                                    lr=learning_rate,
                                    bs=batch_size,
                                    return_losses=True,
                                    event_horizon_time=event_horizon_time,
                                    random_seed=self.random_seed)

    self.torch_model = (model[0].eval(), model[1])
    self.fitted = True
    self.event_horizon_time = event_horizon_time

    return self

  def predict_risk(self, x, t=None):

    if self.fitted:
      return 1-self.predict_survival(x, t)
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_risk`.")

  def predict_survival(self, x, t=None):
    r"""Returns the estimated survival probability at time \( t \),
      \( \widehat{\mathbb{P}}(T > t|X) \) for some input data \( x \).

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    t: list or float
        a list or float of the times at which survival probability is
        to be computed
    Returns:
      np.array: numpy array of the survival probabilites at each time in t.

    """
    if not self.fitted:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")

    x = self._preprocess_test_data(x)

    if t is not None:
      if not isinstance(t, list):
        t = [t]

    scores = predict_survival(self.torch_model, x, t)
    return scores

  def compute_nll(self, x, t, e):
    r"""This function computes the negative log likelihood of the given data.
    In case of competing risks, the negative log likelihoods are summed over
    the different events' type.
    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    t: np.ndarray
        A numpy array of the event/censoring times, \( t \).
    e: np.ndarray
        A numpy array of the event/censoring indicators, \( \delta \).
        \( \delta = r \) means the event r took place.
    Returns:
      float: Negative log likelihood.
    """
    if not self.fitted:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `_eval_nll`.")
    processed_data = self._preprocess_training_data(x, t, e, 0, None, 0)
    _, _, _, x_val, t_val, e_val = processed_data
    x_val, t_val, e_val = x_val,\
        _reshape_tensor_with_nans(t_val),\
        _reshape_tensor_with_nans(e_val)

    loss = float(bce_loss(self.torch_model, x_val, t_val, e_val,
                          self.event_horizon_time).detach().numpy())

    return loss
