# jax-hypernetwork

A simple hypernetwork implementation in [jax](https://github.com/google/jax/) using [haiku](https://github.com/deepmind/dm-haiku).

## Example

In this little demo, we create a linear hypernetwork to parametrise the weights of a multilayer perceptron.
```python
import jax
import jax.numpy as jnp
from jax_hypernetwork import LinearHypernetwork
import haiku as hk
rng = jax.random.PRNGKey(0)

# Create the target network
target_network = hk.transform(lambda x: hk.nets.MLP([10, 10], with_bias=False)(x))
params_target = target_network.init(rng, jnp.empty((28 * 28)))

# Create the hypernetwork
hnet = hk.transform(
    lambda: LinearHypernetwork(params_target, chunk_shape=(1, 10), embedding_dim=7)()
)
params_hnet = hnet.init(rng)

# Use hypernetwork to parametrise the target network
params_target_generated = hnet.apply(params_hnet, rng)
output = target_network.apply(params_target_generated, rng, jnp.empty((28 * 28)))
```

## Install
Install `jax-hypernetwork` using pip:
```
pip install git+https://github.com/smonsays/jax-hypernetwork
```