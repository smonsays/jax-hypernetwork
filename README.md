# jax-hypernetwork

A simple hypernetwork implementation in [jax](https://github.com/google/jax/) using [haiku](https://github.com/deepmind/dm-haiku).

## Example

In this little demo, we create a linear hypernetwork to parametrise the weights of a multilayer perceptron. For a more elaborate example see [example.py](https://github.com/smonsays/jax-hypernetwork/blob/main/example.py).

```python
import haiku as hk
import jax

from jax_hypernetwork import LinearHypernetwork

rng = jax.random.PRNGKey(0)

# 1) Create the target network
target_network = hk.transform(lambda x: hk.nets.MLP([10, 10], with_bias=False)(x))
params_target = target_network.init(rng, jax.numpy.empty((28 * 28)))

# 2) Create the hypernetwork
hnet = hk.transform(
    lambda: LinearHypernetwork(params_target, chunk_shape=(1, 10), embedding_dim=7)()
)
params_hnet = hnet.init(rng)

# 3) Use hypernetwork to parametrise the target network
params_target_generated = hnet.apply(params_hnet, rng)
output = target_network.apply(params_target_generated, rng, jax.numpy.empty((28 * 28)))
```

## Features

The [Hypernetwork base class](https://github.com/smonsays/jax-hypernetwork/blob/main/jax_hypernetwork/hnet.py#L27) allows to specify the

- source model (weight generator)
- target model (model to be parametrised)
- embedding model
- chunking strategy

and is easy to extend e.g. to add input dependence. Two examples provided are a [LinearHypernetwork](https://github.com/smonsays/jax-hypernetwork/blob/main/jax_hypernetwork/hnet.py#L95) and a [MLPHypernetwork](https://github.com/smonsays/jax-hypernetwork/blob/main/jax_hypernetwork/hnet.py#L101).

## Install

Install `jax-hypernetwork` using pip:

```bash
pip install git+https://github.com/smonsays/jax-hypernetwork
```
