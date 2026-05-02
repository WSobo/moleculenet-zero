"""Tests for the Graph class."""
import numpy as np
import pytest

from mnz.graph import Graph
from mnz.layers import GCNLayer
from mnz.layers import Linear
from mnz.layers import mean_readout

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chain_graph():
    """The chain graph 0-1-2-3 with simple scalar features.
    
    Used as the canonical small graph for tests. Each test gets a fresh instance.
    """
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    edge_index = np.array([
        [0, 1, 1, 2, 2, 3],   # source nodes
        [1, 0, 2, 1, 3, 2],   # destination nodes
    ])
    return Graph(X=X, edge_index=edge_index)


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------

def test_chain_graph_basic(chain_graph):
    """Chain 0-1-2-3 should report 4 nodes and 6 directed edges."""
    assert chain_graph.num_nodes == 4
    assert chain_graph.num_edges == 6


def test_chain_graph_adjacency(chain_graph):
    """Adjacency matrix should match a naive reference and the hand-computed expected."""
    # naive reference: build A by looping over edges
    A_naive = np.zeros((chain_graph.num_nodes, chain_graph.num_nodes))
    for k in range(chain_graph.num_edges):
        src, dst = chain_graph.edge_index[:, k]
        A_naive[src, dst] = 1
    
    A_fast = chain_graph.adjacency()
    assert np.array_equal(A_naive, A_fast), "Naive and fast adjacency should match"
    
    expected_A = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ])
    assert np.array_equal(A_fast, expected_A), "Adjacency matrix should match hand-derived"
    assert np.array_equal(A_fast, A_fast.T), "Adjacency should be symmetric for undirected graph"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Normalized adjacency tests
# ---------------------------------------------------------------------------

def test_normalized_adjacency_matches_naive_reference(chain_graph):
    """Optimized normalized_adjacency should match a slow np.diag-based reference."""
    A_tilde = chain_graph.adjacency() + np.eye(chain_graph.num_nodes)
    d_inv_sqrt_diag = np.diag(A_tilde.sum(axis=1) ** -0.5)
    expected = d_inv_sqrt_diag @ A_tilde @ d_inv_sqrt_diag
    
    np.testing.assert_allclose(chain_graph.normalized_adjacency(), expected)


def test_normalized_adjacency_chain_values(chain_graph):
    """Hand-derived values for the chain 0-1-2-3 with self-loops."""
    A_hat = chain_graph.normalized_adjacency()
    # diagonal: self-loop weights, 1/(deg+1) where deg+1 includes the self-loop
    assert np.isclose(A_hat[0, 0], 0.5)              # node 0: deg 1 + self-loop = 2
    assert np.isclose(A_hat[1, 1], 1/3)              # node 1: deg 2 + self-loop = 3
    # off-diagonal: 1 / sqrt(deg_i * deg_j) with self-loop-inflated degrees
    assert np.isclose(A_hat[0, 1], 1 / np.sqrt(6))   # 1 / sqrt(2 * 3)


def test_normalized_adjacency_is_symmetric(chain_graph):
    """A_hat = D^-1/2 (A+I) D^-1/2 should be symmetric."""
    A_hat = chain_graph.normalized_adjacency()
    np.testing.assert_allclose(A_hat, A_hat.T)


# ---------------------------------------------------------------------------
# Layer tests
# ---------------------------------------------------------------------------

def test_gcn_layer_output_shape(chain_graph):
    """GCNLayer should produce output of shape (N, out_features)."""
    gcn = GCNLayer(in_features=1, out_features=2, seed=42)
    output = gcn(chain_graph)
    assert output.shape == (chain_graph.num_nodes, 2), "GCNLayer output shape should be (N, out_features)"
    
def test_linear_layer_output_shape():
    """Linear layer should produce output of shape (N, out_features)."""
    linear = Linear(in_features=1, out_features=3, seed=42)
    X = np.array([[1.0], [2.0], [3.0]])
    output = linear(X)
    assert output.shape == (X.shape[0], 3), "Linear layer output shape should be (N, out_features)"
    
def test_linear_output_shape_2d():
    linear = Linear(in_features=8, out_features=1, seed=42)
    X = np.zeros((4, 8))
    assert linear(X).shape == (4, 1)

def test_linear_output_shape_1d():
    linear = Linear(in_features=8, out_features=1, seed=42)
    x = np.zeros(8)
    assert linear(x).shape == (1,)
    
def test_mean_readout_correctness():
    """mean_readout should average node features correctly."""
    H = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # shape (3, 2)
    graph_vector = mean_readout(H)
    expected = np.array([3.0, 4.0])  # mean of each column
    np.testing.assert_allclose(graph_vector, expected)
    
# ----------------------------------------------------------------------------
# TODO: Add tests once backward propogation is added
# ----------------------------------------------------------------------------