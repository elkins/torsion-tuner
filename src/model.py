import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

class GNNLayer(eqx.Module):
    lin_node: eqx.nn.Linear
    lin_edge: eqx.nn.Linear
    lin_out: eqx.nn.Linear
    
    def __init__(self, node_dim, out_dim, key):
        keys = jr.split(key, 3)
        self.lin_node = eqx.nn.Linear(node_dim, out_dim, key=keys[0])
        self.lin_edge = eqx.nn.Linear(out_dim, out_dim, key=keys[1])
        self.lin_out = eqx.nn.Linear(out_dim, out_dim, key=keys[2])
        
    def __call__(self, x, adj):
        # x: (n_nodes, node_dim)
        # adj: (n_nodes, n_nodes)
        
        h = jax.vmap(self.lin_node)(x)
        
        # Message passing
        # m_i = sum_j adj_ij * h_j
        m = jnp.matmul(adj, h)
        
        # Update
        out = jax.nn.relu(h + m)
        return out

class FineTunerGNN(eqx.Module):
    layers: list
    output_head: eqx.nn.Linear
    
    def __init__(self, node_dim, hidden_dim, out_dim, n_layers, key):
        keys = jr.split(key, n_layers + 1)
        self.layers = []
        curr_dim = node_dim
        for i in range(n_layers):
            self.layers.append(GNNLayer(curr_dim, hidden_dim, key=keys[i]))
            curr_dim = hidden_dim
            
        self.output_head = eqx.nn.Linear(hidden_dim, out_dim, key=keys[-1])
        
    def __call__(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        
        # Predict delta dihedrals
        out = jax.vmap(self.output_head)(x)
        return out
