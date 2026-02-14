# Distributed K-Means with PySpark
This is a custom implementation of the K-Means clustering algorithm built from scratch using PySpark RDDs.

The goal of this project was to understand the parallelization mechanics of K-Means without relying on high-level libraries like Spark MLlib. 

## Architecture & Performance
Architecture: Check architecture_diagram.png to see how the driver and workers communicate.

Speed: See speedup_graph.png. The algorithm scales almost linearly as you add more nodes.

## Setup
### Requirements
You just need Python and PySpark. numpy is used for the vector math, and scikit-learn is only used in the tests to verify our results against the "gold standard."

Bash:
pip install -r requirements.txt

## How to Run
### Quick Start
To run the full suite of tests (correctness, performance, and the new empty-cluster check), just run the script directly:

Bash:
spark-submit kmeans_spark.py

### Using it in your code
Here is the cleanest way to use it.

1. For maximum speed (Production)
If you just want the results and don't care about seeing the error curve drop:

Python:
from pyspark import SparkContext
from kmeans_spark import KMeans

sc = SparkContext()
\# Important: Cache your data! K-Means iterates over it many times.
rdd = sc.parallelize(data).cache()

\# compute_sse=False makes it much faster
kmeans = KMeans(k=5, max_iter=100, compute_sse=False)
kmeans.fit(rdd, sc)

\# Get your results
labels = kmeans.predict(rdd, sc).collect()

2. For debugging/analysis
If you want to tune your k or see if the model is converging:

Python:
\# Enable SSE to track the error rate
kmeans = KMeans(k=5, max_iter=100, compute_sse=True)
kmeans.fit(rdd, sc)

print(f"Error history: {kmeans.sse_history}")

## API Overview
The class is designed to look similar to scikit-learn.

Python:
KMeans(
    k=3,                # Number of clusters
    max_iter=100,       # Stop after this many rounds
    tolerance=1e-4,     # Stop if centroids move less than this
    compute_sse=False   # Set True to track convergence (slower)
)


## Common Issues & Tips
Out of Memory:
    Make sure you aren't collecting massive RDDs to the driver.
    Try increasing the number of partitions: rdd.repartition(100).

Slow convergence:
    K-Means is sensitive to where the centroids start. If it's stuck, try a different seed.
    If your data has very different scales (e.g., age vs income), normalize it first.


## Files included
kmeans_spark.py - The main logic.
architecture_diagram.png - Visual flow.