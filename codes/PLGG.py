import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm


class PointSet:
    def __init__(self, L, sd):
        """
        Initialize a PointSet instance.

        Parameters:
        L (float): The length of the torus (square dimensions are L x L).
        sd (float): The standard deviation of Gaussian noise for perturbation,
                    or 666 to generate a Poisson process.
        """
        self.L = L       # Length of the torus
        self.sd = sd     # Standard deviation for Gaussian perturbation
        
        # Generate points
        self.points = self.generate_points()
        self.distances = self.compute_torus_distances()
    
    def generate_points(self):
        """
        Generates points based on the value of sd:
        - Gaussian-perturbed square lattice for standard deviations other than 666.
        - Poisson process when sd == 666.
        
        Returns:
        np.ndarray: An array of points within the torus dimensions.
        """
        if self.sd == 666:
            # Generate a Poisson process on the torus
            return self.generate_poisson_process()
        else:
            # Generate perturbed lattice points
            return self.generate_perturbed_lattice()
    
    def generate_perturbed_lattice(self):
        """
        Generates a perturbed square lattice on a torus with modular conditions.
        
        Returns:
        np.ndarray: An array of perturbed lattice points within the torus dimensions.
        """
        # Number of lattice points along each axis (distance between nodes is 1)
        n_points = int(self.L)
        
        # Generate initial square lattice points (grid)
        x, y = np.meshgrid(np.arange(n_points), np.arange(n_points))
        x, y = x.ravel(), y.ravel()
        
        # Stack into (N, 2) array for easy manipulation
        points = np.vstack((x, y)).T
        
        # Perturb each point by adding Gaussian noise
        perturbations = np.random.normal(0, self.sd, points.shape)
        perturbed_points = points + perturbations
        
        # Apply modular operation to handle the torus wrapping
        perturbed_points = np.mod(perturbed_points, self.L)
        
        return perturbed_points

    def generate_poisson_process(self):
        """
        Generates a homogeneous Poisson process on the torus.
        
        Returns:
        np.ndarray: An array of Poisson-distributed points within the torus dimensions.
        """
        # Mean number of points proportional to the area of the torus
        mean_points = self.L**2
        
        # Number of points is drawn from a Poisson distribution
        n_points = np.random.poisson(mean_points)
        
        # Generate uniform random points in the torus
        x = np.random.uniform(0, self.L, n_points)
        y = np.random.uniform(0, self.L, n_points)
        
        return np.vstack((x, y)).T
    
    def compute_torus_distances(self):
        """
        Compute the torus distances between all points using the absolute value method.
        
        Returns:
        np.ndarray: A (N, N) distance matrix.
        """
        N = len(self.points)
        
        # Calculate absolute differences
        diff = np.abs(np.expand_dims(self.points, axis=1) - self.points)  # (N, N, 2)
        
        # Apply torus adjustment: minimum of direct difference and wrapped-around difference
        min_diff = np.minimum(diff, self.L - diff)  # (N, N, 2)
        
        # Compute Euclidean distances
        distances = np.linalg.norm(min_diff, axis=2)  # (N, N)
        
        return distances
    
    def visualize(self):
        """
        Visualizes the points on the torus.
        """
        plt.figure(figsize=(6, 6))
        plt.scatter(self.points[:, 0], self.points[:, 1], s=10, color="blue", alpha=0.6)
        plt.xlim(0, self.L)
        plt.ylim(0, self.L)
        plt.title("Point Set on Torus")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def save_distance_matrix(self,filename):
        np.save(filename,self.distances)




class HardGeometricGraph:
    def __init__(self, pointset, radius, torus_length):
        """
        Initialize the Soft Geometric Graph.

        Parameters:
        points (np.ndarray): An (N, 2) array of points in the torus.
        radius (float): The radius scaling factor for connection probabilities.
        torus_length (float): The length of the torus (assumes square torus of size L x L).
        """
        self.points = pointset.points
        self.dis = pointset.distances
        self.radius = radius
        self.torus_length = torus_length
        self.graph = self.construct_graph()
        
    def compute_distances(self, pointset, points):
        """
        Compute the distance matrix based on the provided points or use the one from pointset.
        Parameters:
        pointset (PointSet): An instance of PointSet.
        points (np.ndarray, optional): A custom array of points.
        Returns:
        np.ndarray: The distance matrix for the provided or default points.
        """
        if points is not None:
            # Compute torus distances for the provided points
            N = len(points)
            diff = np.abs(np.expand_dims(points, axis=1) - points)  # (N, N, 2)
            min_diff = np.minimum(diff, pointset.L - diff)  # Handle torus wrapping
            return np.linalg.norm(min_diff, axis=2)  # Compute Euclidean distances
        else:
            # Use the distance matrix from the pointset
            return pointset.distances

    def construct_graph(self):
        """
        Construct the graph by connecting points within the given radius using the distance matrix.
        Returns:
        nx.Graph: The constructed hard geometric graph.
        """
        N = len(self.points)
        G = nx.Graph()

        # Add nodes with their positions
        for i, point in enumerate(self.points):
            G.add_node(i, pos=point)

        # Create a boolean upper triangular adjacency matrix based on distance threshold
        connections = np.triu(self.dis <= self.radius, k=1)  # Only consider upper triangle, exclude diagonal

        # Extract edges from the adjacency matrix
        edges = np.column_stack(np.where(connections))  # Convert True positions to edge indices

        # Add edges to the graph
        G.add_edges_from(edges)

        return G

    def visualize(self):
        """
        Visualize the hard geometric graph.
        """
        pos = nx.get_node_attributes(self.graph, 'pos')  
        edges = self.graph.edges()

        plt.figure(figsize=(8, 8))

        for edge in edges:
            p1, p2 = self.points[edge[0]], self.points[edge[1]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', alpha=0.5)

        plt.scatter(self.points[:, 0], self.points[:, 1], s=30, color="blue")

        plt.title("Hard Geometric Graph")
        plt.xlim(0, self.torus_length)
        plt.ylim(0, self.torus_length)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()


class SoftGeometricGraph:
    def __init__(self, pointset, radius, torus_length):
        """
        Initialize the Soft Geometric Graph.

        Parameters:
        points (np.ndarray): An (N, 2) array of points in the torus.
        radius (float): The radius scaling factor for connection probabilities.
        torus_length (float): The length of the torus (assumes square torus of size L x L).
        """
        self.points = pointset.points
        self.dis = pointset.distances
        self.radius = radius
        self.torus_length = torus_length
        self.adj = self.construct_adj()
        self.graph = self.construct_graph()

    def construct_adj(self):
        """
        Construct the adjacency matrix based on the connection probability formula.

        Returns:1
        np.ndarray: A symmetric adjacency matrix with probabilities.
        """
        N = len(self.points)
        adj = np.zeros((N, N))

        # Avoid division by zero by setting diagonal to infinity
        distances = self.dis.copy()
        np.fill_diagonal(distances, np.inf)

        # Calculate probabilities using vectorized operations
        probabilities = self.radius / (distances**3)
        probabilities = np.clip(probabilities, 0, 1)  # Ensure probabilities are in [0, 1]

        return probabilities

    def construct_graph(self):
        """
        Construct the graph by using the adjacency matrix with connection probabilities.

        Returns:
        nx.Graph: The constructed soft geometric graph.
        """
        N = len(self.points)
        G = nx.Graph()

        # Add nodes
        for i, point in enumerate(self.points):
            G.add_node(i, pos=point)

        # Generate a random matrix with the same shape as adj
        rand_matrix = np.random.rand(N, N)

        # Use upper triangular part of both matrices to decide edges
        connections = np.triu(rand_matrix < self.adj, k=1)  # k=1 ignores the diagonal

        # Extract edges from the upper triangular adjacency matrix
        edges = np.column_stack(np.where(connections))  # Convert True positions to edge indices

        # Add edges to the graph
        G.add_edges_from(edges)

        return G

    def visualize(self):
        """
        Visualize the soft geometric graph.
        """
        pos = nx.get_node_attributes(self.graph, 'pos')  # Get node positions
        edges = self.graph.edges()

        plt.figure(figsize=(8, 8))

        # Plot edges
        for edge in edges:
            p1, p2 = self.points[edge[0]], self.points[edge[1]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', alpha=0.5)

        # Plot nodes
        plt.scatter(self.points[:, 0], self.points[:, 1], s=30, color="blue")

        plt.title("Soft Geometric Graph")
        plt.xlim(0, self.torus_length)
        plt.ylim(0, self.torus_length)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()


def Hard_LCC(L, radii, sds, n_simulations):

    results = {sd: [] for sd in sds}

    for sd in tqdm(sds, desc="模拟不同SD值", position=0):
        sd_fractions = []
        
        for radius in tqdm(radii, desc=f"处理SD={sd}的不同半径", leave=False, position=1):
            fractions = []
            for _ in range(n_simulations):
                points = PointSet(L, sd)
                RGG = HardGeometricGraph(points, radius, L)
                largest_cc = len(max(nx.connected_components(RGG.graph), key=len))
                fractions.append(largest_cc)
                
            mean = np.mean(fractions)
            sd_fractions.append(mean)
        
        results[sd] = sd_fractions

    return results

def Soft_LCC(L, radii, sds, n_simulations):

    results = {sd: [] for sd in sds}

    for sd in tqdm(sds, desc="模拟不同SD值", position=0):
        sd_fractions = []
        
        for radius in tqdm(radii, desc=f"处理SD={sd}的不同半径", leave=False, position=1):
            fractions = []
            for _ in range(n_simulations):
                points = PointSet(L, sd)
                RGG = SoftGeometricGraph(points, radius, L)
                largest_cc = len(max(nx.connected_components(RGG.graph), key=len))
                fractions.append(largest_cc)
                
            mean = np.mean(fractions)
            sd_fractions.append(mean)
        
        results[sd] = sd_fractions

    return results



if __name__ == "__main__":

    L = 5  
    sds = [0 , 0.2, 0.4, 0.6, 0.8, 1, 666]  
    n_simulations = 50  
    radii = np.arange(0, 0.35, 0.01)
    results = Soft_LCC(L, radii, sds, n_simulations)
    plt.figure(figsize=(10, 6))
    for sd, fractions in results.items():
        plt.plot(radii, fractions,marker = '.', label=f"SD={sd}")
    plt.xlabel("Radius")
    plt.ylabel("最大连通分量中的节点占比")
    plt.title("不同SD下最大连通分量占比随Radius变化")
    plt.legend()
    plt.grid()
    plt.show()
    
    print("Soft graph")
    print(results)

#----------------------------------------
    
    radii = np.arange(0.8, 1.4, 0.01)  
    results = Hard_LCC(L, radii, sds, n_simulations)

    plt.figure(figsize=(10, 6))
    for sd, fractions in results.items():
        plt.plot(radii, fractions,marker = '.', label=f"SD={sd}")
    plt.xlabel("Radius")
    plt.ylabel("最大连通分量中的节点占比")
    plt.title("不同SD下最大连通分量占比随Radius变化")
    plt.legend()
    plt.grid()
    plt.show()
    
    print("Hard graph")
    print(results)

    



    