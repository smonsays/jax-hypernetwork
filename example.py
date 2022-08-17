"""
Simple example illustrating how to use a hypernetwork
to partially parametrise a target network.
"""
import jax
import jax.tree_util as jtu
from jax_hypernetwork import LinearHypernetwork
from jax_hypernetwork.utils import dict_filter, flatten_dict, unflatten_dict

import haiku as hk

# Define hyperparameters
input_dim = 128
output_dim = 10
hidden_dim = 50
hidden_layers = 2
embedding_dim = 10


@hk.without_apply_rng
@hk.transform
def target_network(inputs):
    return hk.nets.MLP(output_sizes=hidden_layers * [hidden_dim])(inputs)


# Prepare randomness
rng = jax.random.PRNGKey(0)
rng_input, rng_hnet, rng_target = jax.random.split(rng, 3)
sample_input = jax.random.normal(rng_input, shape=(input_dim,))

# Split params into those to be generated by hnet and those to be direclty optimized
params_all = target_network.init(rng_target, sample_input)
params_target_bias = dict_filter(params_all, "w", all_but_key=True)
params_target_weights = dict_filter(params_all, "w")


@hk.without_apply_rng
@hk.transform
def hnet():
    return LinearHypernetwork(
        params_target=params_target_weights,
        chunk_shape=(1, hidden_dim),
        embedding_dim=embedding_dim,
    )()


# Generate weights using the hypernetwork
params_hnet = hnet.init(rng_hnet)
params_target_weights_generated = hnet.apply(params_hnet)

# Check that PyTreeDef for generated weights matches those of the original params
assert (
    jtu.tree_flatten(params_target_weights)[1]
    == jtu.tree_flatten(params_target_weights_generated)[1]
)

# To use with target_network, combine generated and non-generated params
params_target = unflatten_dict({
    **flatten_dict(params_target_bias),
    **flatten_dict(params_target_weights_generated)
})

sample_output = target_network.apply(params_target, sample_input)
