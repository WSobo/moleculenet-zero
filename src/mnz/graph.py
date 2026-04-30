"""graph.py: Graph class for representing a graph structure for GCN"""
import numpy as np
from dataclasses import dataclass

@dataclass
class Graph:
    """Graph class for representing a graph structure for GCN"""
    X : np.ndarray
    edge_index : np.ndarray

    @property
    def num_nodes(self) -> int:
        return self.X.shape[0]
    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]
    
    def __post_init__(self):
        assert self.X.ndim == 2, "X should be a 2D array"
        assert self.edge_index.ndim == 2, "edge_index should be a 2D array"
        assert self.edge_index.shape[0] == 2, "edge_index should have shape (2, num_edges)"
        assert self.edge_index.max() < self.num_nodes, "edge_index should contain node indicies less than num_nodes"
        assert self.edge_index.min() >= 0, "edge_index should contain node indicies greater than or equal to 0"

    def adjacency(self) -> np.ndarray:
        """Returns the adjacency matrix of the graph using numpy indexing"""
        A = np.zeros((self.num_nodes, self.num_nodes))
        src = self.edge_index[0, :]                   # all source nodes at once, shape (E,)
        dst = self.edge_index[1, :]                   # all destination nodes at once, shape (E,)
        A[src, dst] = 1             # numpy fancy indexing handles all (E,) entries in parallel
        return A
    
    def normalized_adjacency(self) -> np.ndarray:
        """GCN-style normalized adjacency: D̃^(-1/2) (A+I) D̃^(-1/2), shape (N, N)."""
        A_tilde = self.adjacency() + np.eye(self.num_nodes)  # A + I, shape (N, N)
        d_inv_sqrt = A_tilde.sum(axis=1) ** -0.5     # shape (N,)
        return d_inv_sqrt[:, None] * A_tilde * d_inv_sqrt[None, :]
    
    
        







if __name__ == "__main__":
    
    X = np.array([[6,0],[7,1],[6,0],[8,-1]])
    
    edge_index = np.array(                        # edge_index should have shape (2, num_edges), directed from i->j and directed from j->i to make it undirected i <-> j . Each column is one directed edge
        [[0,1,1,2,2,3],      #source nodes
         [1,0,2,1,3,2]])     #destination nodes
    
    graph = Graph(X=X, edge_index=edge_index)
    print("Graph created successfully!")
    
    print("Number of nodes:", graph.num_nodes)
    print("Number of edges:", graph.num_edges)