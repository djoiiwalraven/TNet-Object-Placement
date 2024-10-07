import numpy as np
import random
import matplotlib.pyplot as plt
import sys  # Import sys for flushing the output

# Parameters
GRID_SIZE = 12          # Increased grid size
POPULATION_SIZE = 500   # Increased population size
GENERATIONS = 100       # Increased number of generations
MUTATION_RATE = 1/GRID_SIZE**2    # Mutation rate remains the same
SPRINKLER_RANGE = 2     # Increased sprinkler coverage range
ALIGNMENT_WEIGHT = 50   # New parameter for alignment penalty

# Generate a random floor plan with multiple obstructions
def generate_floor_plan():
    floor_plan = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    # Place walls around the perimeter
    floor_plan[0, :] = 1
    floor_plan[-1, :] = 1
    floor_plan[:, 0] = 1
    floor_plan[:, -1] = 1
    # Add multiple random obstructions
    num_obstructions = GRID_SIZE // 2  # Adjust the number as needed
    for _ in range(num_obstructions):
        obstruction_x = random.randint(2, GRID_SIZE - 3)
        obstruction_y = random.randint(2, GRID_SIZE - 3)
        floor_plan[obstruction_y, obstruction_x] = 1
    return floor_plan

# Generate initial population
def generate_initial_population(size, population_size, floor_plan):
    population = []
    # Optionally, define grid positions to encourage alignment
    # For example, align along columns (vertical alignment)
    grid_columns = np.arange(1, size - 1)  # All columns except the walls
    # Uncomment the following line to select every Nth column (e.g., every 3rd column)
    # grid_columns = np.arange(1, size - 1, 3)

    for _ in range(population_size):
        gene = np.zeros((size, size), dtype=int)
        # Start with fewer sprinklers
        num_sprinklers = random.randint(1, size // 2)
        for _ in range(num_sprinklers):
            # Select positions only from grid columns
            x = random.choice(grid_columns)
            y = random.randint(1, size - 2)
            if floor_plan[y, x] == 0:
                gene[y, x] = 1
        population.append(gene)
    return population

# Calculate coverage array for a gene
def calculate_coverage(floor_plan, gene):
    size = floor_plan.shape[0]
    coverage = np.zeros((size, size), dtype=int)
    for y in range(size):
        for x in range(size):
            if gene[y, x] == 1:
                # Sprinkler found, calculate coverage
                for dy in range(-SPRINKLER_RANGE, SPRINKLER_RANGE + 1):
                    for dx in range(-SPRINKLER_RANGE, SPRINKLER_RANGE + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < size and 0 <= nx < size:
                            if floor_plan[ny, nx] == 0:
                                # Check for obstructions between sprinkler and node
                                if not is_obstructed(floor_plan, x, y, nx, ny):
                                    coverage[ny, nx] += 1
    return coverage

# Check if there is an obstruction between two points
def is_obstructed(floor_plan, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return False
    for step in range(1, steps + 1):
        nx = x1 + int(round(dx * step / steps))
        ny = y1 + int(round(dy * step / steps))
        if floor_plan[ny, nx] == 1:
            return True
    return False

# Fitness function with alignment penalty
def fitness(floor_plan, gene):
    coverage = calculate_coverage(floor_plan, gene)
    uncovered = np.sum((coverage == 0) & (floor_plan == 0))
    overlap = np.sum(coverage[coverage > 1])  # Sum of overlaps (coverage > 1)
    num_sprinklers = np.sum(gene)
    
    # Calculate alignment penalties
    sprinkler_positions = np.argwhere(gene == 1)
    if sprinkler_positions.size == 0:
        # No sprinklers placed; maximum penalty to discourage this gene
        alignment_penalty = 10000
    else:
        # Extract unique rows and columns where sprinklers are placed
        unique_rows = np.unique(sprinkler_positions[:, 0])
        unique_columns = np.unique(sprinkler_positions[:, 1])
        num_unique_rows = len(unique_rows)
        num_unique_columns = len(unique_columns)
        
        # Choose the alignment axis
        # For alignment along one axis only, uncomment the preferred axis
        alignment_axis = 'vertical'  # Align along columns
        # alignment_axis = 'horizontal'  # Align along rows
        
        if alignment_axis == 'horizontal':
            # Penalize the number of unique rows used
            alignment_penalty = num_unique_rows * ALIGNMENT_WEIGHT
        elif alignment_axis == 'vertical':
            # Penalize the number of unique columns used
            alignment_penalty = num_unique_columns * ALIGNMENT_WEIGHT
        else:
            # Penalize the minimum number of unique axes used (either axis)
            min_unique_axes = min(num_unique_rows, num_unique_columns)
            alignment_penalty = min_unique_axes * ALIGNMENT_WEIGHT
    
    # Adjusted penalties
    fitness_value = (
        uncovered * 1000       # Heavier penalty for uncovered nodes
        + overlap * 10         # Increased penalty for overlaps
        + num_sprinklers * 100 # Penalty for the number of sprinklers
        + alignment_penalty    # Penalty for misalignment
    )
    return fitness_value

# Selection function (tournament selection)
def selection(population, fitness_scores):
    selected = []
    for _ in range(len(population)):
        i, j = random.sample(range(len(population)), 2)
        if fitness_scores[i] < fitness_scores[j]:
            selected.append(population[i])
        else:
            selected.append(population[j])
    return selected

# Crossover function (two-point crossover)
def crossover(parent1, parent2):
    size = parent1.shape[0]
    child = np.copy(parent1)
    point1 = random.randint(0, size * size - 1)
    point2 = random.randint(point1, size * size - 1)
    idx = np.unravel_index(range(point1, point2 + 1), parent1.shape)
    child[idx] = parent2[idx]
    return child

# Mutation function
def mutate(gene, mutation_rate,floor_plan):
    size = gene.shape[0]
    # Define grid positions (e.g., to align along columns)
    grid_columns = np.arange(1, size - 1)  # All columns except the walls
    # Uncomment the following line to select every Nth column (e.g., every 3rd column)
    # grid_columns = np.arange(1, size - 1, 3)

    total_positions = len(grid_columns) * (size - 2)
    num_mutations = max(1, int(mutation_rate * total_positions))
    for _ in range(num_mutations):
        x = random.choice(grid_columns)
        y = random.randint(1, size - 2)
        if floor_plan[y, x] == 0:
            gene[y, x] = 1 - gene[y, x]  # Flip between 0 and 1
    return gene

# Main genetic algorithm function with progress display
def genetic_algorithm(floor_plan):
    population = generate_initial_population(GRID_SIZE, POPULATION_SIZE, floor_plan)
    best_fitness = float('inf')
    best_gene = None
    best_generation = 0  # Track when the best fitness was found

    for generation in range(GENERATIONS):
        fitness_scores = [fitness(floor_plan, gene) for gene in population]
        sorted_indices = np.argsort(fitness_scores)
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # Preserve the best individuals (elitism)
        elite_size = max(1, int(0.1 * POPULATION_SIZE))
        new_population = population[:elite_size]

        if fitness_scores[0] < best_fitness:
            best_fitness = fitness_scores[0]
            best_gene = population[0]
            best_generation = generation

        # Print progress on the same line
        print(
            f"\rGeneration: {generation+1}/{GENERATIONS} | "
            f"Current Best Fitness: {best_fitness} (Found at Generation {best_generation+1})",
            end=''
        )
        sys.stdout.flush()

        # Optional: Early stopping if perfect fitness is achieved
        if best_fitness == 0:
            break

        # Selection
        selected_population = selection(population, fitness_scores)
        # Crossover and Mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            child = crossover(parent1, parent2)
            child = mutate(child, MUTATION_RATE,floor_plan)
            new_population.append(child)

        population = new_population

    print()  # Move to the next line after completion
    return best_gene

# Interactive visualization function
def visualize_interactive(floor_plan, gene):
    coverage = calculate_coverage(floor_plan, gene)
    size = floor_plan.shape[0]

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title('Sprinkler Coverage and Placement (Click to Add/Remove Sprinklers)')
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_xticks(np.arange(-0.5, size, 1))
    ax.set_yticks(np.arange(-0.5, size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.invert_yaxis()  # To match array indexing

    # Function to update the plot
    def update_plot():
        ax.clear()
        ax.set_title(f'Fitness: {fitness(floor_plan, gene)}')
        ax.set_xlim(-0.5, size - 0.5)
        ax.set_ylim(-0.5, size - 0.5)
        ax.set_xticks(np.arange(-0.5, size, 1))
        ax.set_yticks(np.arange(-0.5, size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
        ax.invert_yaxis()  # To match array indexing

        coverage = calculate_coverage(floor_plan, gene)
        # Plot floor plan walls and obstructions
        for y in range(size):
            for x in range(size):
                if floor_plan[y, x] == 1:
                    rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black')
                    ax.add_patch(rect)
                else:
                    # Plot coverage
                    if coverage[y, x] > 0:
                        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='lightblue', alpha=0.5)
                        ax.add_patch(rect)
                        # Add coverage number
                        ax.text(x, y, f"{coverage[y, x]}", va='center', ha='center', color='blue', fontsize=6)
        # Plot sprinklers
        sprinkler_y, sprinkler_x = np.where(gene == 1)
        ax.scatter(sprinkler_x, sprinkler_y, color='red', s=50, label='Sprinkler')

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', label='Wall/Obstruction', markerfacecolor='black', markersize=15),
            plt.Line2D([0], [0], marker='s', color='w', label='Coverage Area', markerfacecolor='lightblue', markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', label='Sprinkler', markerfacecolor='red', markersize=15)
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.draw()

    # Event handler for mouse clicks
    def on_click(event):
        if event.inaxes != ax:
            return
        x_click = int(round(event.xdata))
        y_click = int(round(event.ydata))
        if 0 <= x_click < size and 0 <= y_click < size:
            if floor_plan[y_click, x_click] == 0:
                # Toggle sprinkler placement
                gene[y_click, x_click] = 1 - gene[y_click, x_click]
                update_plot()

    # Connect the event handler
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    update_plot()
    plt.show()


