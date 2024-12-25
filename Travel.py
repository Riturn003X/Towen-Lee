import numpy as np
import random
import matplotlib.pyplot as plt


# 计算两点之间的距离（这里简单用二维坐标举例，可以根据实际需求调整）
def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


# 旅行商问题类，包含城市坐标等信息以及相关方法
class TSP:
    def __init__(self, num_cities):
        self.num_cities = num_cities
        self.cities = np.random.rand(self.num_cities, 2)

    # 计算给定路径的总路程（适应度函数，这里取距离的倒数，距离越短适应度越高）
    def fitness(self, path):
        total_distance = 0
        for i in range(len(path)):
            j = (i + 1) % len(path)
            total_distance += distance(self.cities[path[i]], self.cities[path[j]])
        return 1 / total_distance


# 遗传算法类，实现遗传算法的各个步骤
class GeneticAlgorithm:
    def __init__(self, tsp, population_size, mutation_rate, generations):
        self.tsp = tsp
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = self.initialize_population()

    # 初始化种群，生成一组随机的路径排列作为初始种群
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            path = list(range(self.tsp.num_cities))
            random.shuffle(path)
            population.append(path)
        return population

    # 选择操作，使用轮盘赌选择法选择父代个体
    def selection(self):
        fitness_scores = [self.tsp.fitness(path) for path in self.population]
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        selected_population = []
        for _ in range(self.population_size):
            r = random.random()
            cumulative_probability = 0
            for i in range(len(self.population)):
                cumulative_probability += probabilities[i]
                if cumulative_probability >= r:
                    selected_population.append(self.population[i])
                    break
        return selected_population

    # 交叉操作，这里使用顺序交叉（OX）为例
    def crossover(self, parent1, parent2):
        start = random.randint(0, self.tsp.num_cities - 2)
        end = random.randint(start + 1, self.tsp.num_cities - 1)
        child = [-1] * self.tsp.num_cities
        for i in range(start, end + 1):
            child[i] = parent1[i]
        pointer = 0
        for i in range(self.tsp.num_cities):
            if parent2[i] not in child:
                while child[pointer]!= -1:
                    pointer += 1
                child[pointer] = parent2[i]
        return child

    # 变异操作，随机交换两个城市的位置
    def mutation(self, path):
        if random.random() < self.mutation_rate:
            index1, index2 = random.sample(range(self.tsp.num_cities), 2)
            path[index1], path[index2] = path[index2], path[index1]
        return path

    # 运行遗传算法，迭代多代进行优化
    def run(self):
        best_fitness_history = []
        for _ in range(self.generations):
            new_population = []
            selected_population = self.selection()
            for i in range(0, self.population_size, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[i + 1]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_population.extend([child1, child2])
            self.population = new_population
            best_path = max(self.population, key=self.tsp.fitness)
            best_fitness = self.tsp.fitness(best_path)
            best_fitness_history.append(best_fitness)
        return best_path, best_fitness_history

num_cities = 10  # 城市数量，可以根据需求调整
population_size = 100  # 种群大小
mutation_rate = 0.01  # 变异率
generations = 200  # 迭代代数

tsp = TSP(num_cities)
ga = GeneticAlgorithm(tsp, population_size, mutation_rate, generations)
best_path, fitness_history = ga.run()

print("最佳路径:", best_path)
print("最佳路径路程:", 1 / fitness_history[-1])

# 绘制适应度变化曲线
plt.plot(range(generations), fitness_history)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Genetic Algorithm for TSP')
plt.show()
