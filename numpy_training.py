#%%
import math
import random
import numpy as np
#print out cpu time
import time

#%%


# getting a random number
rand = random.random()

#%%
start_time = time.time()

# Python list: sin of 100,000,000 random numbers
pl = [math.sin(random.random()) for i in range(100_000_000)]

end_time = time.time()
execution_time = end_time - start_time

print("Execution time:", execution_time, "seconds")

#%%

start_time = time.time()

na = np.sin(np.random.rand(100_000_000))

end_time = time.time()
execution_time = end_time - start_time

print("Execution time:", execution_time, "seconds")

#%%