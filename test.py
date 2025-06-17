from Practical_Problems_Suite import get_practical_problem
import numpy as np

# Set the dimensionality
dimension:int = 10

# Try to get the Tersoff_potentialC1
problem_instance = get_practical_problem(1122,dim=dimension)
vector_inp = np.random.uniform(-5,5,(dimension,))

result = problem_instance.evaluate(vector_inp)
print(result)

