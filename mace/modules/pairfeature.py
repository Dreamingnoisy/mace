import torch
from torch.autograd import Function

class AOPairFeatures(Function):
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

