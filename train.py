import os
import jax
import jax.numpy as jnp
import optax
import wandb
import data, utils
import transformer
import mlp
from flax import nnx
from tqdm.auto import tqdm
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf.dictconfig import DictConfig
import pickle
from math import prod
from functools import partial
from optax import scale_by_adam
from optax._src import base
from optax import tree_utils as otu
from flax.nnx.training.optimizer import _opt_state_variables_to_state
from utils import eval_step_fn

def scale_by_sqrt_adam_preconditioner(b2: float = 0.95, eps: float = 1e-8):
  def init_fn(_):
    return None
  def update_fn(updates, state, params=None):
    del params
    nu_hat = otu.tree_bias_correction(state.nu, b2, state.count)
    updates = jax.tree.map(
        lambda m, v: None if m is None else m / (jnp.sqrt(v) + eps) ** 0.5,
        updates,
        nu_hat,
        is_leaf=lambda x: x is None,
    )
    return updates, None
  return base.GradientTransformation(init_fn, update_fn)

@nnx.jit
def eval_features_and_logits(features, logits, prev_features, prev_logits):
  delta_features = features - prev_features
  delta_logits = logits - prev_logits
  return {'h': jnp.sqrt(jnp.mean(features**2)), 'dh': jnp.sqrt(jnp.mean(delta_features**2)),
          'f': jnp.sqrt(jnp.mean(logits**2)), 'df': jnp.sqrt(jnp.mean(delta_logits**2))}

def train_and_evaluate(cfg: DictConfig):
  tokens_per_step = cfg.opt.B * cfg.model.L
  if cfg.opt.B_max is None:
    B_micro = cfg.opt.B
  else:
    B_micro = min(cfg.opt.B, cfg.opt.B_max)
    assert cfg.opt.B % B_micro == 0, \
        "`cfg.opt.B` must be an integer multiple of `cfg.opt.B_max`"
  accum_steps = cfg.opt.B // B_micro
  B_eval = cfg.opt.B; tokens_per_step_eval = B_eval * cfg.model.L
  get_batch_train, total_train_tokens = data.make_loader(cfg.ds_path, cfg.model.L, B_micro, 'train', cfg.seed if cfg.shuffle else None)
  try:
    get_batch_test, total_test_tokens = data.make_loader(cfg.ds_path, cfg.model.L, B_eval, 'test', None)
  except:
    get_batch_test, total_test_tokens = data.make_loader(cfg.ds_path, cfg.model.L, B_eval, 'val', None)
  print(f"Total train tokens: {total_train_tokens/1e9:.2f}B")
  print(f"Total test tokens: {total_test_tokens/1e9:.2f}B")

  if os.path.exists(f'{cfg.ds_path}/meta.pkl'):
    # find vocab size
    with open(f'{cfg.ds_path}/meta.pkl', 'rb') as f:
      meta = pickle.load(f)
      cfg.model.V = int(jnp.ceil(meta['vocab_size'] / 32)) * 32

  # model
  mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ('data',))
  # Create the base model and extract its shapes
  if cfg.arch == 'transformer':
    create_model_fn = transformer.create_sharded_model
  elif cfg.arch == 'mlp':
    create_model_fn = mlp.create_sharded_model
  else:
    raise ValueError(f'Unknown model type: {cfg.model.type}')
  base_model_cfg = cfg.model.copy(); base_model_cfg.D = cfg.model.base_D
  base_model = create_model_fn(base_model_cfg, mesh, cfg.seed)
  base_shapes = jax.tree.map(lambda x: jnp.array(x.shape), nnx.state(base_model))
  del base_model
  model = create_model_fn(cfg.model, mesh, cfg.seed)
  shapes = jax.tree.map(lambda x: jnp.array(x.shape), nnx.state(model))
  param_names = jax.tree_util.tree_map_with_path(lambda path, p: '.'.join(str(x) for x in path), nnx.state(model))
  # for ablating mup: match lrs to µP at match_mup_at_D (opt.mup should still be set to true)
  if cfg.model.match_mup_at_D is not None:
    ref_model_cfg = cfg.model.copy(); ref_model_cfg.D = cfg.model.match_mup_at_D
    ref_model = create_model_fn(ref_model_cfg, mesh, cfg.seed)
    # prentend shapes are what they would be at match_mup_at_D
    shapes = jax.tree.map(lambda x: jnp.array(x.shape), nnx.state(ref_model))
    del ref_model
  num_params = sum(prod(p.shape) for p in jax.tree_util.tree_leaves(nnx.state(model)))
  print(f'Number of parameters: {num_params:.2g}')

  # possibly auto-set training horizon (tokens T)
  assert cfg.T is not None or cfg.token_per_param is not None or (cfg.scale is not None and cfg.exponent is not None), \
      "Either T or token_per_param or scale and exponent must be provided."
  if cfg.token_per_param is not None:
    cfg.T = round(cfg.token_per_param * num_params)
    print(f"Setting T to {cfg.T:.2g} from {cfg.token_per_param} toks/param")
  elif cfg.scale is not None and cfg.exponent is not None:
    C = 1e15 * ((num_params / cfg.scale) ** cfg.exponent) # assume compute is measured in petaflops during the fit
    cfg.T = round(C / (6 * num_params))
    print(f"Setting T to {cfg.T} from scale and exponent")
  num_train_steps = cfg.T // tokens_per_step
  num_valid_steps = cfg.T_eval // tokens_per_step_eval
  print(f"Number of train steps: {num_train_steps}")
  print(f"Number of valid steps: {num_valid_steps}")
  
  data_sharding = NamedSharding(mesh, P('data'))
  with mesh: ds_valid = jnp.stack([jax.device_put(next(get_batch_test), data_sharding) for i in range(num_valid_steps)])

  def assign_lr(base_shape, shape, param_name):
      base_lr                   = cfg.opt.lr
      base_din, base_dout       = base_shape
      din, dout                 = shape
      din, dout                 = din / base_din, dout / base_dout
      if   'embed'   in param_name: 
        base_lr *= cfg.opt.embed_lr_mult
      elif 'readout' in param_name: 
        base_lr *= cfg.opt.readout_lr_mult
      if not cfg.opt.mup:
        return base_lr
      mult = 1 / din
      return base_lr * mult

  peak_lrs = jax.tree.map(assign_lr, base_shapes, shapes, param_names)

  # in principle, you should scale eps with 1/dout in µP,
  # but eps is small enough in our experiments so we don't do that.
  # if you keep scaling width, scaling eps may be necessary
  scale_by_optim = scale_by_adam(b1=cfg.opt.b1, b2=cfg.opt.b2, eps=cfg.opt.eps)
  scale_by_peak_lr = optax.GradientTransformation(
      init   = lambda _: None,
      update = lambda upd, state, _: (jax.tree.map(
          lambda g, lr: g * lr, upd, peak_lrs), state)
  )
  scale_by_sqrt_peak_lr = optax.GradientTransformation(
      init   = lambda _: None,
      update = lambda upd, state, _: (jax.tree.map(
          lambda g, lr: g * jnp.sqrt(lr), upd, peak_lrs), state)
  )
  scale_by_adam_and_peak_lr = optax.chain(scale_by_optim, scale_by_peak_lr)
  # Implements P^{-1/2} = diag(peak_lr / (sqrt(v^2) + eps))
  scale_by_sqrt_preconditioner = optax.chain(scale_by_sqrt_adam_preconditioner(b2=cfg.opt.b2, eps=cfg.opt.eps), scale_by_sqrt_peak_lr)

  def scale_by_sqrt_preconditioner_fn(x, opt_state):
     # return (px, new_opt_state). Only want the latter
     return scale_by_sqrt_preconditioner.update(x, opt_state, None)[0]

  schedule_fn = utils.get_scheduler(
      cfg.opt.schedule,
      cfg.opt.decay_frac,
      cfg.opt.warmup_tokens // tokens_per_step,
      num_train_steps
  )

  tx = optax.chain(
      scale_by_adam_and_peak_lr,
      optax.scale_by_schedule(schedule_fn),
      optax.scale(-1.0)
  )

  optimizer = nnx.Optimizer(model, tx)

  # Structure of optimizer state: ((optimizer, lr), lr_schedule, -1)
  def get_optimizer_state(optimizer):
     return _opt_state_variables_to_state(optimizer.opt_state[0]) # want (optimizer, lr)

  del base_shapes, shapes

  get_in_out = partial(data.get_in_out, task='regression' if cfg.ds_path == 'fourier' else 'ntp')
  loss_fn = optax.l2_loss if cfg.ds_path == 'fourier' else optax.softmax_cross_entropy_with_integer_labels

  @nnx.jit
  def batch_loss_fn(model, batch):
    x, y, weights = get_in_out(batch)
    logits = model(x)
    losses = loss_fn(logits, y).mean()
    mean_loss = jnp.sum(losses * weights) / (weights.sum() + 1e-6)
    return mean_loss

  def _micro_loss_and_grad(model, batch):
      return nnx.value_and_grad(batch_loss_fn)(model, batch)

  @nnx.jit
  def train_step(optimizer, micro_batches):
      loss_sum, grads_sum = 0.0, None
      for batch in micro_batches:
          loss, grads = _micro_loss_and_grad(optimizer.model, batch)
          loss_sum += loss
          grads_sum = grads if grads_sum is None else jax.tree.map(jnp.add, grads_sum, grads)

      grads_mean = jax.tree.map(lambda g: g / accum_steps, grads_sum)
      optimizer.update(grads_mean)
      return {'train_loss': loss_sum / accum_steps}

  # Build preconditioned grad_fn
  graphdef, _ = nnx.split(model, nnx.Param)

  def grad_fn(param, batch):
    model = nnx.merge(graphdef, param)
    return nnx.grad(batch_loss_fn)(model, batch)

  def loss_and_grad_fn(param, batch):
    model = nnx.merge(graphdef, param)
    return nnx.value_and_grad(batch_loss_fn)(model, batch)
  
  eval_step = nnx.jit(partial(eval_step_fn, sqrt_P_inv=scale_by_sqrt_preconditioner_fn, loss_and_grad_fn=loss_and_grad_fn))

  # start wandb
  if cfg.wandb_project is not None:
    config = utils.flatten_dict(cfg)
    config['num_params'] = num_params
    config = {k.split('/')[-1]: v for k, v in config.items()}
    wandb.init(project=cfg.wandb_project, config=config, mode=cfg.wandb_mode, tags=[cfg.wandb_tag])

  # training loop
  pending_train_metrics = None
  pending_eval_metrics = None
  tau = 0
  prev_features = None
  prev_logits = None
  pbar = tqdm(range(num_train_steps))
  
  eval_steps = set(
      [int(i) for i in jnp.linspace(0, num_train_steps-1, cfg.num_evals//2)] + 
      [int(i) for i in jnp.geomspace(1, num_train_steps-1, cfg.num_evals//2)]
  )
  eval_steps.add(0)

  with mesh:
    for step in pbar:
      lr_t = cfg.opt.lr * schedule_fn(step)
      tokens_seen = step*tokens_per_step
      compute_spent = 6 * tokens_seen * num_params
      if pending_eval_metrics is not None:
        if cfg.wandb_project is not None: wandb.log(pending_eval_metrics)
        pending_eval_metrics = None

      # eval step at linearly spaced intervals
      if step in eval_steps or ((step+1) == num_train_steps):
        opt_state = get_optimizer_state(optimizer)
        _, params = nnx.split(model, nnx.Param)

        pending_eval_metrics = eval_step(params, ds_valid, opt_state)
        features, logits = model.get_features_and_logits(get_in_out(ds_valid[0])[0])
        if prev_features is not None and prev_logits is not None:
          pending_feature_metrics = eval_features_and_logits(features, logits, prev_features, prev_logits)
        else:
          pending_feature_metrics = {'h': jnp.sqrt(jnp.mean(features**2)), 'f': jnp.sqrt(jnp.mean(logits**2))}
        pending_eval_metrics |= pending_feature_metrics
        prev_features = features
        prev_logits = logits
        pending_eval_metrics |= {'step': step, 'tokens': tokens_seen, 'compute': compute_spent, 'tau': tau, 'lr': lr_t}

      # training step
      micro_batches = [jax.device_put(next(get_batch_train), data_sharding)
                      for _ in range(accum_steps)]
      train_metrics = train_step(optimizer, micro_batches)
      train_metrics |= {'step': step, 'tokens': tokens_seen,
                        'compute': compute_spent, 'tau': tau, 'lr': lr_t}
      tau += lr_t

      # async logging
      if pending_train_metrics is not None:
        pbar.set_postfix_str(f'L={pending_train_metrics["train_loss"]:.2f}')
      pending_train_metrics = train_metrics

    if cfg.wandb_project is not None:
      wandb.log(pending_eval_metrics)
      wandb.finish()