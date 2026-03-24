
from torch.autograd import Function
import torch
from typing import List
class OldAOPairFeatures(Function):
    '''
    Add self-defined autograd function for AoPairFeatures w.r.t. atomic positions
    '''

    @staticmethod
    def forward(ctx,
                positions: torch.Tensor, 
                pair_feature: torch.Tensor,
                pair_feature_grad: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            position: (natm, 3) atomic positions with requires_grad=True
            pair_indices: (npairs, 2) tensor of [atomA, atomB] indices
            pair_feature: (npairs, num_features) precomputed AO feature values
            pair_feature_grad: (npairs, natm, 3, nfeatures) precomputed d(feature)/dR
                
        Returns:
            features: (npairs, nfeatures) tensor
        """

        ctx.save_for_backward(positions)
        ctx.pair_feature_grad = pair_feature_grad
        return pair_feature

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: gradient w.r.t to AoPairFeatures (num_edges, num_features)
        """
        pair_feature_grad = ctx.pair_feature_grad
        grad_input = torch.einsum('pf, paxf->ax', grad_output, pair_feature_grad)

        return grad_input, None, None



class AOPairFeatures(Function):
    '''
    Autograd function for AoPairFeatures w.r.t. atomic positions
    Supports batched graphs with per-graph gradient storage for memory efficiency
    '''

    @staticmethod
    def forward(ctx,
                positions: torch.Tensor, 
                pair_feature: torch.Tensor,
                pair_feature_grad_list: List[torch.Tensor],
                edge_batch: torch.Tensor,
                atom_batch: torch.Tensor,
                ) -> torch.Tensor:
        """
        Args:
            positions: (num_atoms_total, 3) - atomic positions with requires_grad=True
            pair_feature: (num_edges_total, 18) - precomputed AO feature values
            pair_feature_grad_list: List of tensors, one per graph
                                    Each tensor: (num_edges_g, num_atoms_g, 3, 18)
            edge_batch: (num_edges_total,) - graph index for each edge
            atom_batch: (num_atoms_total,) - graph index for each atom
            
        Returns:
            features: (num_edges_total, 18) tensor
        """
        ctx.save_for_backward(positions, edge_batch, atom_batch)
        ctx.num_graphs = len(pair_feature_grad_list)
        ctx.pair_feature_grad_list = pair_feature_grad_list
        return pair_feature

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: gradient w.r.t. AoPairFeatures (num_edges_total, 18)
            
        Returns:
            grad_positions: (num_atoms_total, 3) gradient w.r.t. positions
            None, None, None, None (for other inputs)
        """
        positions, edge_batch, atom_batch = ctx.saved_tensors
        grad_list = ctx.pair_feature_grad_list
        num_graphs = ctx.num_graphs
        
        num_atoms_total = positions.shape[0]
        grad_positions = torch.zeros_like(positions)  # (num_atoms_total, 3)
        
        # Process each graph separately
        for g in range(num_graphs):
            # Get edges and atoms belonging to this graph using boolean masking
            edge_mask = (edge_batch == g)
            atom_mask = (atom_batch == g)
            
            # Skip if no edges or atoms for this graph
            if not edge_mask.any() or not atom_mask.any():
                continue
            
            # Extract indices for this graph
            edge_idx = torch.where(edge_mask)[0]
            atom_idx = torch.where(atom_mask)[0]
            
            # Get gradient tensor for this graph: (num_edges_g, num_atoms_g, 3, 18)
            grad_tensor_g = grad_list[g]
            
            # Get grad_output for this graph's edges: (num_edges_g, 18)
            grad_output_g = grad_output[edge_idx]
            
            # Compute gradient contribution: (num_edges_g, num_atoms_g, 3)
            # grad_output_g: (num_edges_g, 18)
            # grad_tensor_g: (num_edges_g, num_atoms_g, 3, 18)
            grad_atom_g = torch.einsum('ef, eavf->av', 
                                         grad_output_g, 
                                         grad_tensor_g)
            

            
            # Scatter to global positions using atom_idx
            grad_positions.index_add_(0, atom_idx, grad_atom_g)
        
        return grad_positions, None, None, None, None