import math
import numpy as np

# for avoid numerical errors
_EPS = 1e-8
_SIGMA_MAX = 1e32

class CMAES:
  '''CMA-ES algorithm'''
  def __init__(self, mean, sigma, bounds = None, n_max_resampling = 100, seed = 18, population_size=None):
    assert sigma > 0, "sigma must be non-zero positive value"
    n_dim = len(mean)
    if population_size is None:
      ## Calculating population size
      # (eq. 48)
      population_size = 4 + math.floor(3 * math.log(n_dim))  # (eq. 48)
    assert population_size > 0, "popsize must be non-zero positive value."

    # Table 1 caption
    mu = population_size // 2

    ## Calculating weight prime
    # eq. 49
    wts_prime = np.array([
                    math.log((population_size+1) / 2) - math.log(i+1)
                    for i in range(population_size)
                  ])

    # Table 1 caption
    mu_eff = (np.sum(wts_prime[:mu]) ** 2) / ( np.sum(wts_prime[:mu] ** 2) )
    mu_eff_minus = (np.sum(wts_prime[mu:]) ** 2) / ( np.sum(wts_prime[mu:] ** 2) )

    # learning rate for the rank-one update eq. 57
    alpha_cov = 2
    c1 = alpha_cov / ( (n_dim + 1.3)**2 + mu_eff )

    # learning rate for the rank-mu update eq. 58
    alpha_cov = 2
    cmu = min(
          1 - c1 - 1e-8,  # 1e-8 is for large popsize
          alpha_cov * ( ( mu_eff - 2 + (1/mu_eff) ) / ( (n_dim+2)**2 + alpha_cov*(mu_eff/2) ) )
         )
    # From page 30 about "The setting for the default negative weights"
    assert c1 <= 1 - cmu, "invalid learning rate for the rank-one update"
    assert cmu <= 1 - c1, "invalid learning rate for the rank-μ update"

    ## Calculating Wi
    min_alpha = min(
            1 + ( c1/cmu ), # eq. 50
            1 + ( (2*mu_eff_minus)/(mu_eff + 2) ), # eq. 51
            (1 - c1 - cmu) / (n_dim * cmu) # eq.52
            )
    # eq. 53
    positive_sum = np.sum(wts_prime[wts_prime >= 0])
    negative_sum = np.sum(np.abs(wts_prime[wts_prime < 0]))
    weights = np.where(
              wts_prime >= 0,
               (1 / positive_sum) * wts_prime,
               (min_alpha * wts_prime) / negative_sum 
              )

    # eq. 54
    cm = 1

    ## Learning rate for the cumulation for the step-size control
    # eq. 55
    c_sigma = (mu_eff + 2) / (n_dim + mu_eff + 5)
    # damping for sigma
    d_sigma = 1 + 2 * max(0, math.sqrt( (mu_eff-1) / (n_dim+1) ) - 1 ) + c_sigma
    assert c_sigma < 1, "invalid learning rate for cumulation for the step-size control" # eq. 31

    ## Learning rate for cumulation for the rank-one update
    # eq. 56
    cc = (4 + (mu_eff / n_dim)) / (n_dim + 4 + 2*(mu_eff / n_dim))
    assert cc <= 1, "invalid learning rate for cumulation for the rank-one update" # eq. 24
    
    self._n_dim = n_dim
    self._population_size = population_size
    self._mu = mu
    self._mu_eff = mu_eff
    self._cm = cm
    self._cc = cc
    self._c1 = c1
    self._cmu = cmu
    self._c_sigma = c_sigma
    self._d_sigma = d_sigma
    self._weights = weights
    
    # expectation of E||N(0, I)|| == norm(randn(N,1)) (page 28)
    self._chi_N = math.sqrt(self._n_dim) * ( 
              1.0 - (1.0 / (4.0*self._n_dim)) + (1.0 / (21.0*(self._n_dim**2)))
              )
      
    # evolution path (page 29)
    self._p_sigma = np.zeros(self._n_dim)
    self._pc = np.zeros(self._n_dim)
    
    self._mean = mean
    self._C = np.eye(n_dim)
    self._sigma = sigma
    self._D = None
    self._B = None
    
    # bounds contains low and high of each parameter
    assert bounds is None or (mean.size, 2) == bounds.shape, "bounds should be (n_dim, 2)-dim matrix"
    self._bounds = bounds
    
    self._g = 0    # generation number
    self._rng = np.random.RandomState(seed)
    
    self.eigen_eval = 0
    self._n_max_resampling = n_max_resampling

    # Termination criteria (page 33)
    self._tolconditioncov = 1e14  # Indicating condition number of covariance matrix exceeds tolerance
    self._tolxup = 1e4  # Indicating too small sigma or divergent behavior
    self._tolfun = 1e-12  # Indicating range of objective function value is below tolerance
    self._tolx = 1e-12 * sigma  # Indicating standard deviation of normal distribution is smaller than all coordinates

    self._funhist_term = 10 + math.ceil(30 * n_dim / population_size)
    self._funhist_values = np.empty(self._funhist_term * 2)
    
  @property
  def dim(self):
    '''Return the dimension of x of f(x)'''
    return self._n_dim
  
  @property
  def population_size(self):
    '''Return population size'''
    return self._population_size
  
  @property
  def generation(self):
    '''Return the generation number'''
    return self._g
  
  ### FUNCTIONS FOR GENERATING THE POPULATION
  def eigen_decomposition(self) -> (np.ndarray, np.ndarray):
    '''Update B and D from C'''
    # To achieve O(N^2) from page 37 line 80
    if self._B is not None and self._D is not None:
      return self._B, self._D
    
    self._C = (self._C + self._C.T) / 2       # Enforce symmetry page 37 line 82
    #self._C = np.triu(self._C) + np.triu(self._C, 1).T  # Enforce symmetry page 37 line 82
    D2, B = np.linalg.eigh(self._C)
    D = np.sqrt(np.where(D2 < 0, _EPS, D2))  # D contains the standard deviations
    self._C = np.dot(np.dot(B, np.diag(D ** 2)), B.T)
    self._B, self._D = B, D
    '''if ((self._g - self.eigen_eval + 1) > self._population_size*(self._c1+self._cmu) / (10*self._n_dim)):
      self.eigen_eval = self._g + 1
      self._C = np.triu(self._C) + np.triu(self._C, 1).T  # Enforce symmetry page 37 line 82
      D2, B = np.linalg.eigh(self._C)
      D = np.sqrt(np.where(D2 < 0, _EPS, D2))  # D contains the standard deviations
      #self._C = np.dot(np.dot(B, np.diag(D ** 2)), B.T)
      self._B, self._D = B, D
    else:
      B, D = self._B, self._D'''
    return B, D
  
  def sample_solutions(self):
    '''Sampling solutions using the mean and the covariance matrix'''
    B, D = self.eigen_decomposition()
    z = self._rng.randn(self._n_dim)  # ~ N(0, I) eq. 38
    y = B.dot(np.diag(D)).dot(z)  # ~ N(0, C) eq. 39
    x = self._mean + (self._sigma * y)  # ~ N(m, σ^2 C) eq. 40
    return x
  
  def is_feasible(self, param):
    if self._bounds is None:
      return True
    return np.all(param >= self._bounds[:, 0]) and np.all(param <= self._bounds[:, 1])
  
  def repair_infeasible_params(self, param):
    '''Repair the sampled solution belonging to infeasible region'''
    if self._bounds is None:
      return param

    # clip with lower and upper bound.
    param = np.where(param < self._bounds[:, 0], self._bounds[:, 0], param)
    param = np.where(param > self._bounds[:, 1], self._bounds[:, 1], param)
    return param
  
  def ask(self):
    '''Sample parameters for the fitness function'''
    for i in range(self._n_max_resampling):
      # Continue resampling until n_max_resampling until the sampled solution belong to feasible region
      x = self.sample_solutions()
      # Check if the sampled solution belongs to thye feasible region
      if self.is_feasible(x):
        return x
    x = self.sample_solutions()
    x = self.repair_infeasible_params(x)
    return x
  
  # FUNCTIONS FOR UPDATING
  def tell(self, solutions):
    '''
      Function to update the parameters of the distribution
      Tell evaluation values
      solutions = list : [x: np.ndarray, fitness_value: float]
    '''
    if len(solutions) != self._population_size:
      raise ValueError("Must tell popsize-length solutions.")
    
    #B, D = self._B, self._D
    
    self._g += 1

    # SORTING THE SOLUTIONS
    solutions.sort(key=lambda s: s[1], reverse = True) # sorting in descending order for maximization
    print([sol[1] for sol in solutions])

    # Stores 'best' and 'worst' values of the last 'self._funhist_term' generations.
    funhist_idx = 2 * (self.generation % self._funhist_term)
    self._funhist_values[funhist_idx] = solutions[0][1]
    self._funhist_values[funhist_idx + 1] = solutions[-1][1]
    
    # Sample new population of search_points, for k=1, ..., popsize
    B, D = self.eigen_decomposition()
    self._B, self._D = None, None
    
    x_k = np.array([s[0] for s in solutions])  # ~ N(m, σ^2 C)
    y_k = (x_k - self._mean) / self._sigma  # ~ N(0, C) from eq. 15
    
    ## Selection and Recombination
    y_w = np.sum(y_k[:self._mu].T * self._weights[:self._mu], axis=1)  # eq. 41
    self._mean += self._cm * self._sigma * y_w  # eq. 42
    
    ## Step size control
    C_2 = B.dot(np.diag(1 / D)).dot(B.T)  # C^(-1/2) = B D^(-1) B^T page 28
    self._p_sigma = (1 - self._c_sigma) * self._p_sigma + (
            math.sqrt(self._c_sigma * (2 - self._c_sigma) * self._mu_eff
                ) * C_2.dot(y_w)
            )                       # eq. 43
    norm_p_sigma = np.linalg.norm(self._p_sigma)
    self._sigma *= np.exp( (self._c_sigma/self._d_sigma) * ((norm_p_sigma/self._chi_N) - 1) ) # eq. 44
    self._sigma = min(self._sigma, _SIGMA_MAX)
    #self._sigma = min(self._sigma, sys.float_info.max / 5)
    
    ## Covariance matrix adaption
    # For calculating h_sigma (page 28)
    h_sigma_cond_left = norm_p_sigma / math.sqrt(1 - (1 - self._c_sigma) ** (2 * (self._g + 1)))
    h_sigma_cond_right = ( 1.4 + (2 / ( self._n_dim + 1 )) ) * self._chi_N
    h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0

    # Evolution path for rank one update eq.45
    self._pc = (1 - self._cc) * self._pc + (h_sigma * math.sqrt(
          self._cc * (2 - self._cc) * self._mu_eff) * y_w
          )

    # eq.46
    w_io = self._weights * np.where(
      self._weights >= 0,
      1,
      self._n_dim / (np.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + _EPS))

    delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc)  # page 28
    assert delta_h_sigma <= 1  # page 28

    # eq.47
    rank_one = np.outer(self._pc, self._pc)
    rank_mu = np.sum(np.array([w * np.outer(y, y)
                   for w, y in zip(w_io, y_k)]), axis=0)
    self._C = (
      (
        1
        + self._c1 * delta_h_sigma
        - self._c1
        - self._cmu * np.sum(self._weights)
      )
      * self._C
      + (self._c1 * rank_one)
      + (self._cmu * rank_mu)
    )

  def should_stop(self, TolFun = True, TolX = True, TolXUp = True, NoEffectCoord = True, NoEffectAxis = True, ConditionCov = True) -> bool:
    B, D = self.eigen_decomposition()
    dC = np.diag(self._C)

    # Stop if the range of function values of the recent generation is below tolfun.
    if (
      TolFun
      and self.generation > self._funhist_term
      and np.max(self._funhist_values) - np.min(self._funhist_values)
      < self._tolfun
    ):
      print('TolFun criterion executed')
      return True

    # Stop if the std of the normal distribution is smaller than tolx
    # in all coordinates and pc is smaller than tolx in all components.
    if TolX and np.all(self._sigma * dC < self._tolx) and np.all(self._sigma * self._pc < self._tolx):
      print('TolX criterion executed')
      return True

    # Stop if detecting divergent behavior.
    if TolXUp and self._sigma * np.max(D) > self._tolxup:
      print('TolXUP criterion executed')
      return True

    # No effect coordinates: stop if adding 0.2-standard deviations in any single coordinate does not change m.
    if NoEffectCoord and np.any(self._mean == self._mean + (0.2 * self._sigma * np.sqrt(dC))):
      print('NoEffectCoord criterion executed')
      return True

    # No effect axis: stop if adding 0.1-standard deviation vector in
    # any principal axis direction of C does not change m. "pycma" check
    # axis one by one at each generation.
    i = self.generation % self.dim
    if NoEffectAxis and np.all(self._mean == self._mean + (0.1 * self._sigma * D[i] * B[:, i])):
      print('NoEffectAxis criterion executed')
      return True

    # Stop if the condition number of the covariance matrix exceeds 1e14.
    condition_cov = np.max(D) / np.min(D)
    if ConditionCov and (condition_cov > self._tolconditioncov):
      print('ConditionCov criterion executed')
      return True

    return False
