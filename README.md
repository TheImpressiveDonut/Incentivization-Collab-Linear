# Incentivization for Collaborative learning with Linear Estimator

We explore incentivization in the context of collaborative learning, for single round training and for linear models. 
We use exclusively numpy in our experiments. 

### Notations

- $N$ the number of clients, $d$ the dimension of data samples, $m_i$ the number of samples at client $i$ 
- $\beta_i^*$ ground truth, dimension $d \times 1$
- $X_i$ data samples, dimension $m_i \times d$
- $y_i$ observations, dimension $m_i \times 1$
- $\hat{\beta}_i$ local estimate, dimension $d \times 1$
- $\mathbb{E}\hat{\beta}_i$ expected value of local estimate, dimension $d \times 1$
- $\mathcal{W}$ mixing matrix, dimension $n \times n$
- $\hat{\beta}_i^{\text{MTL}}$ aggregated estimation, dimension $d \times 1$
- $\mathcal{U}$ utility function

## Concept

### Loss

Our experiments use exclusively *MSE* to measure the performance of the computed estimator:
$$\text{MSE}(\hat{\beta}_i) = \mathbb{E}\left\lVert \hat{\beta}_i - \beta_i^* \right\rVert_2^2$$
$$\text{MSE}(\hat{\beta}_i^{\text{MTL}}) = \mathbb{E}\left\lVert \hat{\beta}_i^{\text{MTL}} - \beta_i^* \right\rVert_2^2$$

### Aggregate estimations

With the mixing matrix $\mathcal{W}$, we compute the aggregate estimation:

```math
\hat{\beta}_i^{\text{MTL}} = \sum_{i=1}^N \mathcal{W}_{ij} \hat{\beta}_i
```

Such that:
$$\text{MSE}(\hat{\beta}_i^{\text{MTL}}) \leq \text{MSE}(\hat{\beta}_i)$$

### Mixing matrix

The mixing matrix is computed such that it minimize $\text{MSE}(\hat{\beta}_i^{\text{MTL}})$, in theory $\mathcal{W}$ is
computed as such:
```math
\mathcal{W} = K(C + V)^{-1}, \quad K = \left[\left\langle \beta_i^* , \mathbb{E}\hat{\beta}_j \right\rangle\right]_{ij}, 
\quad C = \left[\left\langle \mathbb{E}\hat{\beta}_i , \mathbb{E}\hat{\beta}_j \right\rangle\right]_{ij}, 
\quad V = \text{diag}\left(\left[\mathbb{E}\left\lVert \hat{\beta}_i - \mathbb{E}\hat{\beta}_i \right\rVert_2^2\right]_{ij}\right)
```

In practice, we don't have the such value so try to estimate the mixing matrix: @TODO show algorithm

### Utilities function



## Experiments

### MNIST

We start with the example of MNIST to have nice visualisation, with a simple linear regression:

- Image are of size $28 \times 28$, we transform them into numpy array of size (1, 784), $d = 784$
- $X_i$ of size $m_i$ is the number of image of client $i$
- $\hat{\beta}_i = (X_i^T X_i)^{-1} X_i^T y_i$
- 