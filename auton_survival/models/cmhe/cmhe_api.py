from .cmhe_torch import CoxMixtureHeterogenousEffects
from .cmhe_torch import DeepCoxMixtureHeterogenousEffects
from .cmhe_utilities import train_cmhe, predict_survival

import torch
import numpy as np

class DeepCoxMixturesHeterogenousEffects:
  """A Deep Cox Mixtures with Heterogenous Effects model.

  This is the main interface to a Deep Cox Mixture with Heterogenous Effects.
  A model is instantiated with approporiate set of hyperparameters and
  fit on numpy arrays consisting of the features, event/censoring times
  and the event/censoring indicators.

  For full details on Deep Cox Mixture, refer to the paper [1].

  References
  ----------
  [1] <a href="https://arxiv.org/abs/2101.06536">Deep Cox Mixtures
  for Survival Regression. Machine Learning in Health Conference (2021)</a>

  Parameters
  ----------
  k: int
      The number of underlying base survival phenotypes.
  g: int
      The number of underlying treatment effect phenotypes.
  layers: list
      A list of integers consisting of the number of neurons in each
      hidden layer.
  Example
  -------
  >>> from auton_survival import CoxMixturesHeterogenousEffects
  >>> model = CoxMixturesHeterogenousEffects()
  >>> model.fit(x, t, e, a)

  """

  def __init__(self, layers=None):

    self.layers = layers
    self.fitted = False

  def __call__(self):
    if self.fitted:
      print("A fitted instance of the CMHE model")
    else:
      print("An unfitted instance of the CMHE model")

    print("Hidden Layers:", self.layers)

  def _preprocess_test_data(self, x, a):
    return torch.from_numpy(x).float(), torch.from_numpy(a).float()

  def _preprocess_training_data(self, x, t, e, a, vsize, val_data, random_state):

    idx = list(range(x.shape[0]))

    np.random.seed(random_state)
    np.random.shuffle(idx)

    x_train, t_train, e_train, a_train = x[idx], t[idx], e[idx], a[idx]

    x_train = torch.from_numpy(x_train).float()
    t_train = torch.from_numpy(t_train).float()
    e_train = torch.from_numpy(e_train).float()
    a_train = torch.from_numpy(a_train).float()

    if val_data is None:

      vsize = int(vsize*x_train.shape[0])
      x_val, t_val, e_val, a_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:], a_train[-vsize:]

      x_train = x_train[:-vsize]
      t_train = t_train[:-vsize]
      e_train = e_train[:-vsize]
      a_train = a_train[:-vsize]

    else:

      x_val, t_val, e_val, a_val = val_data

      x_val = torch.from_numpy(x_val).float()
      t_val = torch.from_numpy(t_val).float()
      e_val = torch.from_numpy(e_val).float()
      a_val = torch.from_numpy(a_val).float()

    return (x_train, t_train, e_train, a_train,
    	      x_val, t_val, e_val, a_val)

  def _gen_torch_model(self, inputdim, optimizer):
    """Helper function to return a torch model."""
    if len(self.layers):
      return DeepCoxMixtureHeterogenousEffects(inputdim, layers=self.layers,
                                               optimizer=optimizer)
    else:
      return CoxMixtureHeterogenousEffects(inputdim, optimizer=optimizer)

  def fit(self, x, t, e, a, vsize=0.15, val_data=None,
          iters=1, learning_rate=1e-3, batch_size=100,
          optimizer="Adam", random_state=100):

    r"""This method is used to train an instance of the DSM model.

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    t: np.ndarray
        A numpy array of the event/censoring times, \( t \).
    e: np.ndarray
        A numpy array of the event/censoring indicators, \( \delta \).
        \( \delta = 1 \) means the event took place.
    a: np.ndarray
        A numpy array of the treatment assignment indicators, \( a \).
        \( a = 1 \) means the individual was treated.
    vsize: float
        Amount of data to set aside as the validation set.
    val_data: tuple
        A tuple of the validation dataset. If passed vsize is ignored.
    iters: int
        The maximum number of training iterations on the training dataset.
    learning_rate: float
        The learning rate for the `Adam` optimizer.
    batch_size: int
        learning is performed on mini-batches of input data. this parameter
        specifies the size of each mini-batch.
    optimizer: str
        The choice of the gradient based optimization method. One of
        'Adam', 'RMSProp' or 'SGD'.
    random_state: float
        random seed that determines how the validation set is chosen.
    """

    processed_data = self._preprocess_training_data(x, t, e, a,
                                                    vsize, val_data,
                                                    random_state)

    x_train, t_train, e_train, a_train, x_val, t_val, e_val, a_val = processed_data

    #Todo: Change this somehow. The base design shouldn't depend on child

    inputdim = x_train.shape[-1]

    model = self._gen_torch_model(inputdim, optimizer)

    model, _ = train_cmhe(model,
                          (x_train, t_train, e_train, a_train),
                          (x_val, t_val, e_val, a_val),
                          epochs=iters,
                          lr=learning_rate,
                          bs=batch_size,
                          return_losses=True)

    self.torch_model = (model[0].eval(), model[1])
    self.fitted = True

    return self

  def predict_risk(self, x, a, t=None):

    if self.fitted:
      return 1-self.predict_survival(x, a, t)
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_risk`.")

  def predict_survival(self, x, a, t=None):
    r"""Returns the estimated survival probability at time \( t \),
      \( \widehat{\mathbb{P}}(T > t|X) \) for some input data \( x \).

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    a: np.ndarray
        A numpy array of the treatmeant assignment, \( a \).
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

    x = self._preprocess_test_data(x, a)

    if t is not None:
      if not isinstance(t, list):
        t = [t]

    scores = predict_survival(self.torch_model, x, a, t)
    return scores

