import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad
from scipy.special import i0

def torus_distance(x1, y1, x2, y2, L):
    """
    Calculate the shortest distance between two points on a torus.
    
    Parameters:
    - x1, y1: Coordinates of the first point.
    - x2, y2: Coordinates of the second point.
    - w: Width of the torus.
    - h: Height of the torus.
    
    Returns:
    - The shortest distance between the two points on the torus.
    """
    # Calculate distances in the x and y directions, considering wrapping
    dx = min(abs(x2 - x1), L - abs(x2 - x1))
    dy = min(abs(y2 - y1), L - abs(y2 - y1))
    
    # Return the Euclidean distance
    return (dx**2 + dy**2)**0.5

class PointSet:
    def __init__(self, L, sd):
        """
        Initialize a square lattice of points on a torus with size LxL and then perturb them with Gaussian noise.

        Parameters:
        - L: The length of the sides of the square lattice; determines the number of points on the torus.
        - sd: Standard deviation of the Gaussian perturbations to apply to each coordinate.
        """
        self.L = L
        self.sd = sd
        self.points = self.initialize_points()

    def initialize_points(self):
        """
        Create a square lattice of points on a torus, then perturb each point with Gaussian noise and wrap them around using modulo.

        Returns:
        - A 2D numpy array containing the perturbed and wrapped points.
        """
        # Generate grid points
        x = np.arange(self.L)
        y = np.arange(self.L)
        xv, yv = np.meshgrid(x, y)

        # Flatten the arrays to make a list of coordinates
        points = np.vstack([xv.ravel(), yv.ravel()]).T

        # Apply Gaussian perturbation
        noise = np.random.normal(0, self.sd, points.shape)
        perturbed_points = points + noise

        # Wrap points around the torus using modulo operation
        perturbed_points = np.mod(perturbed_points, self.L)

        return perturbed_points

    def get_points(self):
        """
        Return the current list of points on the torus.

        Returns:
        - A 2D numpy array of the points.
        """
        return self.points

    def save_points(self, filename):
        """
        Save the current points to a file.

        Parameters:
        - filename: The name of the file to save the points to.
        """
        np.save(filename, self.points)

    def load_points(self, filename):
        """
        Load points from a file.

        Parameters:
        - filename: The name of the file from which to load the points.
        """
        self.points = np.load(filename)

class SoftPLGG:
    def __init__(self, point_set, radius):
        """
        Initialize the Soft PLGG model with a given point set and radius.

        Parameters:
        - point_set: An instance of PointSet containing the points on the torus.
        - radius: The radius parameter that influences connection probabilities.
        """
        self.point_set = point_set
        self.radius = radius
        self.connection_matrix = self.calculate_probabilities()

    def calculate_probabilities(self):
        """
        Calculate the connection probabilities for each pair of points based on the Soft PLGG model.

        Returns:
        - A 2D numpy array of connection probabilities.
        """
        points = self.point_set.get_points()
        n = points.shape[0]
        prob_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                D = self.torus_distance(points[i], points[j])
                prob = min(1, self.radius / (D**3))
                prob_matrix[i, j] = prob
                prob_matrix[j, i] = prob  # The matrix is symmetric
        
        return prob_matrix

    def torus_distance(self, p1, p2):
        """
        Calculate the torus distance between two points.
        """
        dx = min(abs(p2[0] - p1[0]), self.point_set.L - abs(p2[0] - p1[0]))
        dy = min(abs(p2[1] - p1[1]), self.point_set.L - abs(p2[1] - p1[1]))
        return (dx**2 + dy**2)**0.5

    def generate_graph(self):
        """
        Generate a real graph based on the connection probabilities.
        """
        n = self.connection_matrix.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < self.connection_matrix[i, j]:
                    G.add_edge(i, j)
        
        return G


    def graph_properties(self, graph):
        """
        Compute properties of the graph: largest component size, mean degree, and degree variance.

        Returns:
        - A dictionary with properties.
        """
        largest_cc = len(max(nx.connected_components(graph), key=len))
        degrees = np.array([d for n, d in graph.degree()])
        mean_degree = np.mean(degrees)
        variance_degree = np.var(degrees)
        
        return {
            'largest_component_size': largest_cc,
            'mean_degree': mean_degree,
            'variance_degree': variance_degree
        }

def experiment(start, end, num_radius, sd, montecarlo, L):
    radii = np.linspace(start, end, num_radius)
    results = {radius: {'largest_component_size': [], 'mean_degree': [], 'variance_degree': []} for radius in radii}

    # 进行 Monte Carlo 模拟
    for _ in tqdm(range(montecarlo // 10), desc="Monte Carlo Simulations"):
        point_sets = [PointSet(L, sd) for _ in range(10)]
        for radius in radii:
            local_results = {'largest_component_size': [], 'mean_degree': [], 'variance_degree': []}
            for point_set in point_sets:
                soft_plgg = SoftPLGG(point_set, radius)
                for _ in range(10):
                    graph = soft_plgg.generate_graph()
                    props = soft_plgg.graph_properties(graph)
                    for key in local_results:
                        local_results[key].append(props[key])
            # 计算每个半径的平均值
            for key in results[radius]:
                results[radius][key].append(np.mean(local_results[key]))

    # 对结果取平均
    for radius in radii:
        for key in results[radius]:
            results[radius][key] = np.mean(results[radius][key])

    # 绘图
    for key in ['largest_component_size', 'mean_degree', 'variance_degree']:
        plt.figure(figsize=(10, 6))
        plt.plot(radii, [results[radius][key] for radius in radii], label=f'Average {key}')
        plt.xlabel('Radius')
        plt.ylabel(key.replace('_', ' ').title())
        plt.title(f'Average {key.replace("_", " ").title()} vs Radius')
        plt.legend()
        plt.grid(True)
        plt.show()

def P(R, lam, sigma):
    """
    Calculate the probability P given R, lambda, and sigma.

    Parameters:
    - R : float
        The upper limit of the integral.
    - lam : float
        The lambda parameter in the formula.
    - sigma : float
        The sigma parameter in the formula.

    Returns:
    - float
        The calculated probability.
    """

    def integrand(t, lam, sigma):
        exponent = -(1/2) * ((t + lam) / (2 * sigma**2))
        bessel = i0(np.sqrt(2 * t * lam) / (2 * sigma))
        return np.exp(exponent) * bessel

    # Perform the numerical integration
    result, _ = quad(integrand, 0, R**2, args=(lam, sigma))
    return 1/2 * result




experiment(start=0.02, end=0.35, num_radius=10, sd=0.1, montecarlo=100, L=10)