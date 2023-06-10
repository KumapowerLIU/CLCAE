import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(
            self,
            opts,
            embedding_dim,

    ):
        super().__init__()
        projection_dim = opts.projection_dim if opts.projection_dim is not None else 512
        dropout = opts.projection_dropout if opts.projection_dropout is not None else 0.1
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class Projection(nn.Module):
    """
    Creates projection head
    Args:
      n_in (int): Number of input features
      n_hidden (int): Number of hidden features
      n_out (int): Number of output features
      use_bn (bool): Whether to use batch norm
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 use_bn: bool = True):
        super().__init__()

        # No point in using bias if we've batch norm
        self.lin1 = nn.Linear(n_in, n_hidden, bias=not use_bn)
        self.bn = nn.BatchNorm1d(n_hidden) if use_bn else nn.Identity()
        self.relu = nn.ReLU()
        # No bias for the final linear layer
        self.lin2 = nn.Linear(n_hidden, n_out, bias=False)

    def forward(self, x):
        x = self.lin1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x