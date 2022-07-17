import lifelines
import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from sklearn.utils import shuffle

from tqdm import tqdm

from auton_survival.models.dsm.utilities import get_optimizer, _reshape_tensor_with_nans


def bce_loss(model, x, t, e, time):
  device = next(model.parameters()).device

  # get rid of event == 0, time-to-event < time
  mask = (e == 0) & (t < time)
  x = x[~mask].to(device).to(torch.float32)
  e = e[~mask].to(device).to(torch.float32)
  t = t[~mask].to(device).to(torch.float32)

  # positive class: time-to-event > time
  y = torch.zeros((x.shape[0]), dtype=torch.float).to(device)
  positive = (t > time)
  y[positive] = 1
  y = y.view(-1, 1)

  logits = model.forward(x)

  # bce loss
  # criterion = nn.BCEWithLogitsLoss(reduction='none')
  # loss = criterion(logits, y)
  # loss = torch.sum(loss)

  criterion = nn.BCEWithLogitsLoss(reduction='none')
  mask = ~torch.isnan(y)

  loss = torch.zeros_like(y)
  loss[mask] = criterion(logits[mask], y[mask])
  loss = torch.nansum(loss)

  return loss


def train_step(model, x, t, e, optimizer, time, bs=256, seed=100):

  x, t, e = shuffle(x, t, e, random_state=seed)

  n = x.shape[0]

  batches = (n // bs) + 1

  epoch_loss = 0

  for i in range(batches):

    xb = x[i*bs:(i+1)*bs]
    tb = t[i*bs:(i+1)*bs]
    eb = e[i*bs:(i+1)*bs]

    # Training Step
    torch.enable_grad()
    optimizer.zero_grad()
    loss = bce_loss(model, xb,
                    _reshape_tensor_with_nans(tb),
                    _reshape_tensor_with_nans(eb),
                    time)

    loss.backward()
    optimizer.step()

    epoch_loss += float(loss)

  return epoch_loss/n


def test_step(model, x, t, e, time):

  with torch.no_grad():
    loss = float(bce_loss(model, x, t, e, time))

  return loss/x.shape[0]


def fit_km_estimators(model, out, t, e,):

  # out = model(x)
  t = np.reshape(t, (-1, 1))
  e = np.reshape(e, (-1, 1))
  out = out.detach().cpu().numpy()
  quantiles = [(1. / model.n_bins) * i for i in range(model.n_bins + 1)]
  outbins = np.quantile(out, quantiles)

  score_conditional_km_estimators = {}

  for n_bin in range(model.n_bins):

    binmin = outbins[n_bin]
    binmax = outbins[n_bin + 1]

    scorebin = (out >= binmin) & (out <= binmax)

    if n_bin == 0:
      binmin = -np.inf
    if n_bin == model.n_bins-1:
      binmax = np.inf
    score_conditional_km_estimators[(binmin, binmax)] = lifelines.KaplanMeierFitter().fit(t[scorebin],
                                                                                          e[scorebin])

  return score_conditional_km_estimators


def train_binary_survival(model, train_data, val_data,
                          event_horizon_time=10*365.25, epochs=50,
                          patience=3, bs=256, lr=1e-3, debug=False,
                          random_seed=0, return_losses=False):

  torch.manual_seed(random_seed)
  np.random.seed(random_seed)

  if val_data is None:
    val_data = train_data

  xt, tt, et = train_data
  xv, tv, ev = val_data

  tt_ = _reshape_tensor_with_nans(tt)
  et_ = _reshape_tensor_with_nans(et)
  tv_ = _reshape_tensor_with_nans(tv)
  ev_ = _reshape_tensor_with_nans(ev)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  optimizer = get_optimizer(model, lr)

  valc = np.inf
  patience_ = 0

  losses = []

  for epoch in tqdm(range(epochs)):

    # train_step_start = time.time()
    _ = train_step(model, xt, tt_, et_, optimizer, event_horizon_time, bs, seed=epoch)
    # print(f'Duration of train-step: {time.time() - train_step_start}')
    # test_step_start = time.time()
    valcn = test_step(model, xv, tv_, ev_, event_horizon_time)
    # print(f'Duration of test-step: {time.time() - test_step_start}')

    losses.append(valcn)

    if epoch % 1 == 0:
      if debug: print(patience_, epoch, valcn)

    if valcn > valc:
      patience_ += 1
    else:
      patience_ = 0

    if patience_ == patience:
      if model.survival_estimator == 'km':
        model = (model,
                fit_km_estimators(model,
                                  model(xt),
                                  tt_.detach().cpu().numpy(),
                                  et_.detach().cpu().numpy())
                )
      else:
        raise NotImplementedError('No estimator named', model.survival_estimator)

      if return_losses:
        return model, losses
      else:
        return model

    valc = valcn

  if model.survival_estimator == 'km':
    model = (model,
            fit_km_estimators(model,
                              model(xt),
                              tt_.detach().cpu().numpy(),
                              et_.detach().cpu().numpy())
            )
  else:
    raise NotImplementedError('No estimator named', model.survival_estimator)

  if return_losses:
    return model, losses
  else:
    return model


def predict_survival(model, x, t_horizons):
  model, score_conditional_km_estimators = model

  if isinstance(t_horizons, (int, float)): t_horizons = [t_horizons]

  out = model(x).detach().cpu().numpy().ravel()
  output = np.zeros((out.shape[0], len(t_horizons)))

  if model.survival_estimator == 'km':
    for (binmin, binmax) in score_conditional_km_estimators:
      mask = (out >= binmin) & (out <= binmax)
      output[mask, :] = score_conditional_km_estimators[(binmin,
                                                         binmax)].predict(t_horizons).T.values
  else:
    raise NotImplementedError('No estimator named', model.survival_estimator)

  return output
