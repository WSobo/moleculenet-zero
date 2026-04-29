import numpy as np
import pytest
from mnz.graph import Graph


def test_chain_graph_basic():
    """Chain 0-1-2-3 should report 4 nodes and 6 directed edges."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    edge_index = np.array([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    g = Graph(X=X, edge_index=edge_index)
    assert g.num_nodes == 4
    assert g.num_edges == 6
    
def test_chain_graph_adjacency():
    """Chain 0-1-2-3 should have correct adjacency matrix."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    edge_index = np.array([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    g = Graph(X=X, edge_index=edge_index)
    A_naive = g.naive_adjacency()
    A_fast = g.adjacency()
    assert np.array_equal(A_naive, A_fast), "Naive and fast adjacency should match"
    expected_A = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ])
    assert np.array_equal(A_fast, expected_A), "Adjacency matrix should match expected"
    # test symmetry for undirected graph
    assert np.array_equal(A_fast, A_fast.T), "Adjacency matrix should be symmetric for undirected graph"
    
def test_rejects_1d_X():
    with pytest.raises(AssertionError):
        Graph(X=np.array([1.0, 2.0]), edge_index=np.array([[0], [1]]))

def test_rejects_1d_edge_index():
    with pytest.raises(AssertionError):
        Graph(X=np.array([[1.0], [2.0]]), edge_index=np.array([0, 1]))

def test_rejects_3xE_edge_index():
    with pytest.raises(AssertionError):
        Graph(X=np.array([[1.0], [2.0]]), edge_index=np.array([[0, 1], [1, 0], [0, 1]]))

def test_rejects_out_of_bounds_edge_index():
    with pytest.raises(AssertionError):
        Graph(X=np.array([[1.0], [2.0]]), edge_index=np.array([[0, 1], [2, 3]]))
        
def test_rejects_negative_edge_index():
    with pytest.raises(AssertionError):
        Graph(X=np.array([[1.0], [2.0]]), edge_index=np.array([[0, 1], [-1, 0]]))

### Tests for normalized adjacency ###

def test_normalized_adjacency_matches_naive_reference(chain_graph):
    """Optimized normalized_adjacency should match a slow np.diag-based reference."""
    A_tilde = chain_graph.adjacency() + np.eye(chain_graph.num_nodes)
    d_inv_sqrt_diag = np.diag(A_tilde.sum(axis=1) ** -0.5)
    expected = d_inv_sqrt_diag @ A_tilde @ d_inv_sqrt_diag
    
    np.testing.assert_allclose(chain_graph.normalized_adjacency(), expected)


def test_normalized_adjacency_chain_values(chain_graph):
    """Hand-derived values for the chain 0-1-2-3."""
    A_hat = chain_graph.normalized_adjacency()
    # diagonal: self-loop weights, 1/(deg+1)
    assert np.isclose(A_hat[0, 0], 0.5)        # node 0: deg 2 with self-loop
    assert np.isclose(A_hat[1, 1], 1/3)        # node 1: deg 3 with self-loop
    # off-diagonal: 1/sqrt(deg_i * deg_j)
    assert np.isclose(A_hat[0, 1], 1 / np.sqrt(6))   # 1/sqrt(2*3)


def test_normalized_adjacency_is_symmetric(chain_graph):
    A_hat = chain_graph.normalized_adjacency()
    np.testing.assert_allclose(A_hat, A_hat.T)
    
    
### tests for forward pass ###