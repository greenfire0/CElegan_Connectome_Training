import tracemalloc
from memory_profiler import profile

@profile
def my_function():
    tracemalloc.start()

    my_list = [i for i in range(10000)]
    my_dict = {i: i for i in range(10000)}

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("Top 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)

    tracemalloc.stop()

if __name__ == "__main__":
    my_function()
