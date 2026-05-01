import numpy as np
from mnz.graph import Graph


class GCNLayer:
    
    def __init__(self, in_features: int, out_features: int, seed: int | None =None):
        self.in_features : int = in_features
        self.out_features : int = out_features
        rng = np.random.default_rng(seed=seed) # create a random number generator instance
        self.W = rng.normal(loc=0.0, scale=np.sqrt(2.0 / self.in_features), size=(self.in_features, self.out_features)) # Weights matrix. He/Kaiming initialization for ReLU activation
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, graph: Graph) -> np.ndarray:
        """Forward pass of the GCN layer"""
        assert graph.X.shape[1] == self.in_features, (f"Expected input features {self.in_features}, got {graph.X.shape[1]}")
        
        A_norm = graph.normalized_adjacency() # shape (N, N)
        X = graph.X # shape (N, self.in_features)
        W = self.W # shape (self.in_features, self.out_features)
        next_layer = A_norm @ X @ W # shape (N, self.out_features)
        return np.maximum(next_layer, 0) # ReLU activation

class Linear:
    def __init__(self, in_features: int, out_features: int, seed: int | None =None):
        self.in_features : int = in_features
        self.out_features : int = out_features
        rng = np.random.default_rng(seed=seed) # create a random number generator instance
        self.W = rng.normal(loc=0.0, scale=np.sqrt(2.0 / self.in_features), size=(self.in_features, self.out_features)) # Weights matrix. He/Kaiming initialization for ReLU activation
        self.b = np.zeros((self.out_features,)) # Bias vector, initialized to zeros

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass of the linear layer"""
        assert X.shape[-1] == self.in_features, (f"Expected input features {self.in_features}, got {X.shape[-1]}")
        
        W = self.W # shape (self.in_features, self.out_features)
        return X @ W + self.b # shape (N, self.out_features)

def mean_readout(H: np.ndarray) -> np.ndarray:
    """Average node features to get a single graph-level vector. (N, F) → (F,)"""
    return H.mean(axis=0)
    
if __name__ == "__main__":
    # the chain graph 0-1-2-3 with 3 input features per node
    X = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.2],
        [1.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],
    ])
    edge_index = np.array([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    g = Graph(X=X, edge_index=edge_index)
    
    layer = GCNLayer(in_features=3, out_features=8, seed=0)
    H1 = layer.forward(g)
    print(f"Input shape:  {g.X.shape}")
    print(f"Output shape: {H1.shape}")
    print(f"Output:\n{H1}")
    
    # determinism check
    H1_again = layer.forward(g)
    print(f"Same on repeat call: {np.array_equal(H1, H1_again)}")
    
    
    # test linear layer
    gcn1 = GCNLayer(in_features=3, out_features=16, seed=0)
    gcn2 = GCNLayer(in_features=16, out_features=16, seed=1)
    classifier = Linear(in_features=16, out_features=1, seed=2)
    
    H1 = gcn1.forward(g)                            # (N, 16)
    g1 = Graph(X=H1, edge_index=g.edge_index)       # wrap H1 back into a Graph for layer 2
    H2 = gcn2.forward(g1)                           # (N, 16)
    graph_embedding = mean_readout(H2)              # (16,)
    logit = classifier.forward(graph_embedding)     # (1,)
    print(f"Logit: {logit}")