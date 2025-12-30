"""
Gravitational Search Algorithm (GSA) for Feature Selection

This module implements GSA for automatic feature selection in the IoT network
attack detection preprocessing pipeline. GSA treats feature subsets as particles
with masses determined by their fitness (classification accuracy).

Author: Nguyen Duc Thang
Project: IoT Network Attack Detection using Federated Learning
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GravitationalSearchAlgorithm:
    """
    Gravitational Search Algorithm for feature selection.
    
    Particles represent binary feature subsets. Better solutions (higher accuracy)
    have stronger gravitational pull on other particles.
    """
    
    def __init__(
        self,
        num_features: int,
        target_num_features: int = 20,
        population_size: int = 30,
        max_iterations: int = 50,
        gravitational_constant: float = 100.0,
        alpha: float = 20.0,
        random_seed: int = 42
    ):
        """
        Initialize GSA parameters.
        
        Args:
            num_features: Total number of features in dataset
            target_num_features: Desired number of features to select
            population_size: Number of candidate solutions  
            max_iterations: Maximum iterations for GSA
            gravitational_constant: Initial gravitational constant G
            alpha: Decay rate for gravitational constant
            random_seed: Random seed for reproducibility
        """
        self.num_features = num_features
        self.target_num_features = target_num_features
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.G_initial = gravitational_constant
        self.alpha = alpha
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        # GSA state
        self.population = None
        self.velocities = None
        self.fitness_history = []
        self.best_solution = None
        self.best_fitness = 0.0
        
        logger.info(f"GSA initialized: {num_features} features → {target_num_features} target")
    
    def initialize_population(self) -> np.ndarray:
        """
        Initialize population with random binary feature subsets.
        
        Returns:
            Population array of shape [population_size, num_features]
        """
        # Initialize with random binary vectors
        population = np.random.rand(self.population_size, self.num_features)
        
        # Convert to binary based on probability
        # Adjust probability to get close to target number of features
        prob = self.target_num_features / self.num_features
        population = (population < prob).astype(int)
        
        # Ensure each individual has at least min_features
        for i in range(self.population_size):
            num_selected = population[i].sum()
            if num_selected < 5:  # Minimum 5 features
                # Randomly select features to add
                zero_indices = np.where(population[i] == 0)[0]
                add_count = 5 - num_selected
                add_indices = np.random.choice(zero_indices, add_count, replace=False)
                population[i, add_indices] = 1
        
        return population
    
    def evaluate_fitness(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_mask: np.ndarray
    ) -> float:
        """
        Evaluate fitness of a feature subset using validation accuracy.
        
        Args:
            X: Feature matrix [num_samples, num_features]
            y: Labels [num_samples]
            feature_mask: Binary mask indicating selected features
            
        Returns:
            Fitness score (validation accuracy)
        """
        # Get selected features
        selected_indices = np.where(feature_mask == 1)[0]
        
        if len(selected_indices) == 0:
            return 0.0  # No features selected
        
        X_selected = X[:, selected_indices]
        
        try:
            # Train-validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X_selected, y, test_size=0.2, random_state=self.random_seed, stratify=y
            )
            
            # Train lightweight proxy model (KNN for speed)
            # KNN is much faster than RandomForest while providing good feature discrimination
            clf = KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1  # Parallel processing
            )
            clf.fit(X_train, y_train)
            
            # Compute validation accuracy
            y_pred = clf.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            # Penalty for deviating from target number of features
            num_selected = len(selected_indices)
            penalty = abs(num_selected - self.target_num_features) / self.num_features
            fitness = accuracy - (0.1 * penalty)  # Small penalty
            
            return max(0.0, fitness)
            
        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            return 0.0
    
    def calculate_masses(self, fitness_scores: np.ndarray) -> np.ndarray:
        """
        Calculate gravitational masses based on fitness scores.
        
        Better solutions have larger masses.
        
        Args:
            fitness_scores: Fitness values for population
            
        Returns:
            Normalized masses
        """
        # Avoid division by zero
        best_fitness = fitness_scores.max()
        worst_fitness = fitness_scores.min()
        
        if best_fitness == worst_fitness:
            return np.ones(self.population_size) / self.population_size
        
        # Calculate masses (normalized fitness)
        masses = (fitness_scores - worst_fitness) / (best_fitness - worst_fitness)
        
        # Normalize to sum to 1
        mass_sum = masses.sum()
        if mass_sum > 0:
            masses = masses / mass_sum
        else:
            masses = np.ones(self.population_size) / self.population_size
        
        return masses
    
    def calculate_forces(
        self,
        population: np.ndarray,
        masses: np.ndarray,
        G: float
    ) -> np.ndarray:
        """
        Calculate gravitational forces acting on each particle.
        
        Args:
            population: Current population positions
            masses: Gravitational masses
            G: Gravitational constant
            
        Returns:
            Total forces for each particle [population_size, num_features]
        """
        forces = np.zeros_like(population, dtype=float)
        
        for i in range(self.population_size):
            total_force = np.zeros(self.num_features)
            
            for j in range(self.population_size):
                if i != j:
                    # Euclidean distance
                    distance = np.linalg.norm(population[i] - population[j]) + 1e-10
                    
                    # Gravitational force: F = G * (M_i * M_j) / distance
                    force_magnitude = G * masses[i] * masses[j] / distance
                    
                    # Direction: from particle i towards particle j
                    direction = population[j] - population[i]
                    
                    # Accumulate force
                    total_force += force_magnitude * direction
            
            forces[i] = total_force
        
        return forces
    
    def update_velocities_and_positions(
        self,
        velocities: np.ndarray,
        population: np.ndarray,
        forces: np.ndarray,
        masses: np.ndarray,
        iteration: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update particle velocities and positions.
        
        Args:
            velocities: Current velocities
            population: Current positions
            forces: Acting forces
            masses: Particle masses
            iteration: Current iteration number
            
        Returns:
            Updated (velocities, population)
        """
        new_velocities = np.zeros_like(velocities)
        new_population = np.zeros_like(population)
        
        for i in range(self.population_size):
            # Acceleration = Force / Mass
            acceleration = forces[i] / (masses[i] + 1e-10)
            
            # Update velocity: V(t+1) = rand * V(t) + A(t)
            rand_factor = np.random.rand(self.num_features)
            new_velocities[i] = rand_factor * velocities[i] + acceleration
            
            # Update position (continuous): X(t+1) = X(t) + V(t+1)
            continuous_position = population[i] + new_velocities[i]
            
            # Convert to binary using sigmoid
            sigmoid_pos = 1 / (1 + np.exp(-continuous_position))
            
            # Threshold to binary
            binary_position = (sigmoid_pos > 0.5).astype(int)
            
            # Ensure minimum number of features
            num_selected = binary_position.sum()
            if num_selected < 5:
                zero_indices = np.where(binary_position == 0)[0]
                if len(zero_indices) > 0:
                    add_count = min(5 - num_selected, len(zero_indices))
                    add_indices = np.random.choice(zero_indices, add_count, replace=False)
                    binary_position[add_indices] = 1
            
            new_population[i] = binary_position
        
        return new_velocities, new_population
    
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Run GSA feature selection.
        
        Args:
            X: Feature matrix [num_samples, num_features]
            y: Labels [num_samples]
            verbose: Print progress
            
        Returns:
            Best feature mask (binary array)
        """
        logger.info(f"Starting GSA with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Initialize population and velocities
        self.population = self.initialize_population()
        self.velocities = np.zeros_like(self.population, dtype=float)
        
        # Main GSA loop
        for iteration in range(self.max_iterations):
            # Evaluate fitness for all particles
            fitness_scores = np.array([
                self.evaluate_fitness(X, y, self.population[i])
                for i in range(self.population_size)
            ])
            
            # Track best solution
            best_idx = fitness_scores.argmax()
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_solution = self.population[best_idx].copy()
            
            self.fitness_history.append({
                'iteration': iteration,
                'best_fitness': self.best_fitness,
                'avg_fitness': fitness_scores.mean(),
                'num_features_best': self.best_solution.sum() if self.best_solution is not None else 0
            })
            
            if verbose and (iteration % 5 == 0 or iteration == self.max_iterations - 1):
                logger.info(
                    f"Iteration {iteration}/{self.max_iterations}: "
                    f"Best Fitness={self.best_fitness:.4f}, "
                    f"Avg Fitness={fitness_scores.mean():.4f}, "
                    f"Features={self.best_solution.sum()}"
                )
            
            # Calculate masses
            masses = self.calculate_masses(fitness_scores)
            
            # Calculate gravitational constant (decays over time)
            G = self.G_initial * np.exp(-self.alpha * iteration / self.max_iterations)
            
            # Calculate forces
            forces = self.calculate_forces(self.population, masses, G)
            
            # Update velocities and positions
            self.velocities, self.population = self.update_velocities_and_positions(
                self.velocities, self.population, forces, masses, iteration
            )
        
        logger.info(f"GSA completed: Best fitness={self.best_fitness:.4f}, "
                   f"Features selected={self.best_solution.sum()}")
        
        return self.best_solution
    
    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """
        Plot GSA convergence curve.
        
        Args:
            save_path: Path to save the figure (optional)
        """
        if not self.fitness_history:
            logger.warning("No fitness history to plot")
            return
        
        iterations = [h['iteration'] for h in self.fitness_history]
        best_fitness = [h['best_fitness'] for h in self.fitness_history]
        avg_fitness = [h['avg_fitness'] for h in self.fitness_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
        plt.plot(iterations, avg_fitness, 'r--', linewidth=1, label='Average Fitness')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Fitness (Validation Accuracy)', fontsize=12)
        plt.title('GSA Convergence Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results(
        self,
        feature_names: List[str],
        save_path: str
    ) -> Dict:
        """
        Save GSA results to JSON.
        
        Args:
            feature_names: Names of all features
            save_path: Path to save JSON file
            
        Returns:
            Results dictionary
        """
        if self.best_solution is None:
            raise ValueError("No solution found. Run GSA first.")
        
        selected_indices = np.where(self.best_solution == 1)[0]
        selected_features = [feature_names[i] for i in selected_indices]
        
        results = {
            'selected_features': selected_features,
            'selected_indices': selected_indices.tolist(),
            'num_selected': len(selected_features),
            'num_original': self.num_features,
            'target_num_features': self.target_num_features,
            'best_fitness': float(self.best_fitness),
            'converged_iteration': len(self.fitness_history),
            'max_iterations': self.max_iterations,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'population_size': self.population_size,
                'gravitational_constant': self.G_initial,
                'alpha': self.alpha,
                'random_seed': self.random_seed
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")
        return results


def load_selected_features(json_path: str) -> List[str]:
    """
    Load selected features from GSA results JSON.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        List of selected feature names
    """
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results['selected_features']


if __name__ == '__main__':
    # Test GSA on a simple dataset
    from sklearn.datasets import make_classification
    
    logger.info("Testing GSA on synthetic dataset")
    
    # Generate synthetic data: 100 samples, 30 features, 5 informative
    X, y = make_classification(
        n_samples=500,
        n_features=30,
        n_informative=10,
        n_redundant=10,
        n_classes=5,
        random_state=42
    )
    
    # Run GSA
    gsa = GravitationalSearchAlgorithm(
        num_features=30,
        target_num_features=10,
        population_size=20,
        max_iterations=30,
        random_seed=42
    )
    
    best_features = gsa.run(X, y, verbose=True)
    
    print(f"\n{'='*60}")
    print(f"GSA Test Complete")
    print(f"{'='*60}")
    print(f"Features selected: {best_features.sum()} / {len(best_features)}")
    print(f"Best fitness: {gsa.best_fitness:.4f}")
    print(f"Selected feature indices: {np.where(best_features == 1)[0]}")
    
    # Plot convergence
    gsa.plot_convergence()
    
    print(f"\n✅ GSA module test passed!")
