import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

K = [1, 2, 3, 4]
seeds = [0, 1, 2, 3, 4] 

# K-means

costs_kmeans = [0]*5
mixtures_kmeans = [0]*5
posts_kmeans = [0]*5

for k in range(len(K)):
    for i in range(len(seeds)):
        
        mixtures_kmeans[i], posts_kmeans[i], costs_kmeans[i] = \
        kmeans.run(X, *common.init(X, K[k], seeds[i]))
    
    print("Cluster: ", k+1, "- Cost: ", np.min(costs_kmeans))


# EM

mixtures_em = [0]*5
posterior_em = [0]*5
likelihood_em = [0]*5

for k in range(len(K)):
    for i in range(len(seeds)):

        mixtures_em[i], posterior_em[i], likelihood_em[i] = \
        naive_em.run(X, *common.init(X, K[k], seeds[i]))

    print("K: ", k+1, "- Maximum log-likelihood: ", np.max(likelihood_em))