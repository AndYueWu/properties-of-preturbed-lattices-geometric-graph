import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

def construct_adj(dis, radius):
    """
    Construct the adjacency matrix based on the connection probability formula.

    Parameters:
    dis (np.ndarray): Distance matrix.
    radius (float): Radius parameter for connection probability.

    Returns:
    np.ndarray: A symmetric adjacency matrix with probabilities.
    """
    distances = dis.copy()
    np.fill_diagonal(distances, np.inf)  # Avoid division by zero
    probabilities = radius / (distances**3)
    probabilities = np.clip(probabilities, 0, 1)  # Ensure probabilities are in [0, 1]
    return probabilities

def simulate_graph(adjacency_matrix):
    """
    Construct a graph based on the adjacency matrix with probabilistic connections.

    Parameters:
    adjacency_matrix (np.ndarray): Matrix with connection probabilities.

    Returns:
    nx.Graph: A constructed graph.
    """
    N = adjacency_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))

    upper_tri_indices = np.triu_indices(N, k=1)
    rand_vals = np.random.rand(len(upper_tri_indices[0]))
    edges = [
        (i, j) for idx, (i, j) in enumerate(zip(*upper_tri_indices))
        if rand_vals[idx] < adjacency_matrix[i, j]
    ]
    G.add_edges_from(edges)
    return G

def analyze_graph(G):
    """
    Analyze a graph to compute fraction of nodes in the largest component,
    mean degree, and variance of degree.

    Parameters:
    G (nx.Graph): The graph to analyze.

    Returns:
    tuple: Fraction of nodes in the largest component, mean degree, variance of degree.
    """
    largest_component = max(nx.connected_components(G), key=len)
    fraction_largest = len(largest_component) / G.number_of_nodes()
    degrees = [deg for _, deg in G.degree()]
    mean_degree = np.mean(degrees)
    variance_degree = np.var(degrees)
    return fraction_largest, mean_degree, variance_degree

def run_simulation(distances_list, radii, n_simulations):
    """
    Run simulations and analyze results for multiple distance matrices.

    Parameters:
    distances_list (list): List of distance matrices.
    radii (np.ndarray): Array of radius values to test.
    n_simulations (int): Number of simulations for each configuration.

    Returns:
    dict: Results for each distance matrix and radius.
    """
    results = {i: {'fractions': [], 'mean_degrees': [], 'variance_degrees': []}
               for i in range(len(distances_list))}

    for i, distances in enumerate(tqdm(distances_list, desc="Processing distance matrices")):
        for radius in tqdm(radii, desc=f"Processing radii for distance {i+1}", leave=False):
            fractions, mean_degrees, variance_degrees = [], [], []
            for _ in range(n_simulations):
                adj_matrix = construct_adj(distances, radius)
                G = simulate_graph(adj_matrix)
                fraction, mean_degree, variance_degree = analyze_graph(G)
                fractions.append(fraction)
                mean_degrees.append(mean_degree)
                variance_degrees.append(variance_degree)
            results[i]['fractions'].append(np.mean(fractions))
            results[i]['mean_degrees'].append(np.mean(mean_degrees))
            results[i]['variance_degrees'].append(np.mean(variance_degrees))

    return results

def plot_results(radii, results, save_path):
    """
    Plot the results for fraction in largest component, mean degree, and variance of degree.

    Parameters:
    radii (np.ndarray): Radius values.
    results (dict): Results from the simulation.
    save_path (str): Path to save the plots and data.
    """
    metrics = ['fractions', 'mean_degrees', 'variance_degrees']
    titles = [
        "Fraction of Nodes in Largest Component",
        "Mean Degree",
        "Variance of Degree"
    ]
    ylabels = ["Fraction", "Mean Degree", "Variance"]
    
    for metric, title, ylabel in zip(metrics, titles, ylabels):
        plt.figure(figsize=(10, 6))
        for i, result in results.items():
            plt.plot(radii, result[metric], label=f"Distance Matrix {i+1}")
        plt.xlabel("Radius")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/{metric}.png")
        plt.close()

    # Save raw data
    np.save(f"{save_path}/results.npy", results)

if __name__ == "__main__":
    # Parameters
    distance_files = [
        "/Users/yuewu/Documents/PLGG/L=100_1.npy",
        "/Users/yuewu/Documents/PLGG/L=100_2.npy",
        "/Users/yuewu/Documents/PLGG/L=100_3.npy"
    ]
    radii = np.arange(0, 0.35, 0.01)
    n_simulations = 100
    save_path = "./simulation_results"

    # Load distances
    distances_list = [np.load(file) for file in distance_files]

    # Run simulations
    results = run_simulation(distances_list, radii, n_simulations)

    # Plot and save results
    plot_results(radii, results, save_path)
