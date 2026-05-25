import jax.numpy as jnp
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts, RANDOM_COIL_CA
from src.data import RESIDUE_TYPES

def get_residue_rc_shifts(res_indices):
    """Map residue indices to their random coil CA shifts."""
    rc_list = [RANDOM_COIL_CA[RESIDUE_TYPES[i]] for i in res_indices]
    return jnp.array(rc_list)

def montelione_loss(phi, psi, target_shifts, res_indices):
    """
    A loss function inspired by the Montelione group's approach of 
    validating structures against experimental chemical shifts.
    """
    rc_shifts = get_residue_rc_shifts(res_indices)
    pred_shifts = predict_ca_shifts(phi, psi, rc_shifts)
    
    # MSE between predicted and experimental CA shifts
    return jnp.mean((pred_shifts - target_shifts)**2)

def calculate_ansurr_proxy(phi, psi):
    """
    A simplified proxy for the ANSURR score, measuring structural 
    consistency with secondary structure propensity.
    """
    # High quality structures should have phi/psi clusters in Ramachandran allowed regions
    # This is a 'soft' penalty for being in disallowed regions
    pass
