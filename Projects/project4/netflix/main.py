import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

K = [1, 2, 3, 4]
seeds = [0, 1, 2, 3, 4] 

# K-means

costs_kmeans = [0]*len(seeds)
mixtures_kmeans = [0]*len(seeds)
posts_kmeans = [0]*len(seeds)

for k in range(len(K)):
    for i in range(len(seeds)):
        
        mixtures_kmeans[i], posts_kmeans[i], costs_kmeans[i] = \
        kmeans.run(X, *common.init(X, K[k], seeds[i]))
    
    print("Cluster: ", k+1, "- Cost: ", np.min(costs_kmeans))


# EM

mixtures_em = [0]*len(seeds)
posterior_em = [0]*len(seeds)
likelihood_em = [0]*len(seeds)
best_likelihood_em = [0]*len(K)
best_mixtures_em = [0]*len(K)

for k in range(len(K)):
    for i in range(len(seeds)):

        mixtures_em[i], posterior_em[i], likelihood_em[i] = \
        naive_em.run(X, *common.init(X, K[k], seeds[i]))

    # Saves the best likelihood values to use in BIC part
    best_likelihood_idx = np.argmax(likelihood_em)
    best_likelihood_em[k] = likelihood_em[best_likelihood_idx]
    best_mixtures_em[k] = mixtures_em[best_likelihood_idx]

    print("K:", k+1, "- Maximum log-likelihood: ", best_likelihood_em[k])

# BIC

bic = [0]*len(K)

for k in range(len(K)):
        bic[k] = common.bic(X, best_mixtures_em[k], best_likelihood_em[k])

        print("Bic for K:", k+1, "= ", bic[k], ".Log likelihood= ", best_likelihood_em[k])
        print("Best K=",np.argmax(bic)+1)
