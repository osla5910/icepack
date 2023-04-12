import scipy
from scipy import stats, special
import numpy as np

class MultiDirichletPrior:
    '''
    Attributes:
        classes
        members
        class_counts
        mixing_rates
        alphas
    '''
    def __init__(self, samples):
        # store the shape of the samples
        self.N, self.J = self.samples.shape
        # self.classes is a boolean array: each row is a class indicating which x_i are present
        self.classes = np.unique(self.samples > 0, axis=0)
        self.num_classes = len(self.classes)

        # self.members is a boolean array:
        #       each row is a sample; each column is a class.
        #       True represents that the sample belongs to the class.
        self.members = np.all((self.samples > 0)[:,:,None] == self.classes.T[None, :, :], axis=1)
        # count how many samples are in each class
        self.class_counts = self.members.sum(axis=0)
        # use these to initialize the mixing ratios for the prior {π_i}
        self.mixing_rates = self.class_counts / self.class_counts.sum()
        # estimate the Dirichlet parameters for each sample
        self.alphas = np.zeros((self.num_classes, self.J))
        for i in range(self.num_classes):
            # print(self.members)
            # raise ValueError
            # print(self.samples[self.members.T[i]][:,self.classes[i]])
            class_samples = self.samples[self.members.T[i]][:,self.classes[i]]
            alpha = self.dirichlet_map(class_samples)
            self.alphas[i][self.classes[i]] = alpha
        # alphas = [self.dirichlet_map(self.samples[membership, c]) for membership,c in zip(self.members.T,self.classes)]
        # self.alphas = np.array(alphas)

    @staticmethod
    def dirichlet_map(samples):
        '''
        Infer dirichlet parameters
        Assume a prior over the α_i of e^(-α_i/λn)
        '''
        # geo_mean = np.exp(np.mean(np.log(samples), axis=0))  # geometric means of the x_i
        # m_est = geo_mean / geo_mean.sum()  # normalize to make the sums 1
        # divergence = scipy.stats.entropy(m_est[None, :], samples, axis=1).mean()  # computes the kullback-leibler divergence of two distributions
        # alpha_est = (J - 1) / divergence
        # alpha_est /= 2
        # alpha_est = min(max(alpha_est, 100), 4)
        # return alpha_est * geo_mean
        gammaln, digamma, polygamma = scipy.special.gammaln, scipy.special.digamma, scipy.special.polygamma
        λ = 100 * len(samples)
        log_xi = np.log(samples).mean(axis=0)
        alpha_0 = samples.mean(axis=0) * 10
        alpha = alpha_0
        f = lambda alpha: gammaln(alpha.sum()) - gammaln(alpha).sum() + (log_xi * (alpha - 1)).sum()
        alphas = [alpha_0]
        for _ in range(15):
            alpha = alphas[-1]
            grad = digamma(alpha.sum()) - digamma(alpha) + log_xi - 1 / λ
            hessian = - np.eye(len(alpha)) * (polygamma(1, alpha))
            # print(alpha)
            hessian += polygamma(1, alpha.sum())
            invH = np.linalg.inv(hessian)
            da = - np.dot(invH, grad)
            alphas.append(alpha + da)
        return alpha
        # like = np.array([f(a) for a in alphas])
        # like /= like[-1]
        # magn = np.array([sum(a) for a in alphas])
        # magn /= magn[-1]
        # plt.plot(like, label='log likelihood')
        # plt.plot(magn, label='magnitude $\sum_i \\alpha_i$')
        # plt.legend()

#%%
class SampleEnsemble:
    '''
    Attributes:
        prior_samples
        posterior_samples
        ice_prior
        y_observed / w_observed
        k_samples_prior
        k_samples_posterior
    '''
    def __init__(self, a_samples, v_samples, h_bnd):
        assert len(a_samples) == len(v_samples)
        self.N = len(a_samples)
        self.num_ice_cats = len(h_bnd) - 1 # there are 5 thickness classes (not including open water)
        self.l= np.empty((self.N, self.num_ice_cats))      # how much ice is on the left side of the interval
        self.r= np.empty((self.N, self.num_ice_cats))      # how much ice is on the right side of the interval
                                                    # for now we don't have r_0, the fraction of open water
                                
        for i in range(self.num_ice_cats):
            M = np.array([[1.,                1. ],
                          [h_bnd[i],  h_bnd[i+1] ]])
            a_i, v_i = a_samples[:,i], v_samples[:,i]
            out = (np.linalg.inv(M) @ np.array([ a_i, v_i ]))
            self.l[:,i], self.r[:,i] = out

        self.w = 1 - a_samples.sum(axis=1)
        self.w = self.w.reshape(-1, 1)

        self.x = np.column_stack((self.w, self.l, self.r))
        self.J = self.x.shape[1]


        self.prior = MultiDirichletPrior(self.x)
        self.N, self.J = self.prior.N, self.prior.J
        self.k_samples_prior = self.prior.members.argmax(axis=1)
        #
        # self.k_uniforms = self.class_to_uniform(self.conditioned_class_probs(), self.k_samples_prior)
        # self.x_uniforms = np.zeros_like(self.samples_prior)
        # self.build_x_uniform()
        #
        # self.w_obs = None
        # self.y_obs = None
        #
        # self.k_samples_post = np.zeros_like(self.k_samples_prior)
        # self.samples_post = np.zeros_like(self.samples_prior)

    def update(self, w_obs):
        self.w_obs = w_obs
        # w is just a shorthand for x_0
        self.samples_post[:,0] = self.w_obs
        # k_uniforms was found by taking the CDF of the class of the sample
        # here we plug that into the invCDF for   k+ | w+
        self.k_samples_post = self.uniform_to_class(self.conditioned_class_probs(prior=False), self.k_uniforms)
        # now we can use our uniforms to build up samples from our dirichlet
        self.build_x_post()

    def conditioned_class_probs(self, prior=True):
        '''
        We assume that samples are generated by drawing w-=x0 then k- then x1 x2 ...
        k- | w-  =  (w- | k-) (k-) / (w-)
        '''
        prop = self.class_likelihoods(prior=prior) * self.prior.mixing_rates[None, :]
        return prop / prop.sum(axis=1, keepdims=1)

    def x_to_uniforms(self, j):
        '''
        x_j | x_i for all i<j
        Let I = Sum x_i
        x_j / (1-I) is distributed as Beta(u_j, Sum u_k) where k>j
        '''
        # the pdf of [x_j|x_i] is a sum of scaled betas
        # with a delta at 0 and 1-I

        for i,x in enumerate(self.x):
            # find the conditional class probabilities [k|x_i] = [x_i|k] • [k-]
            # 


        # find the probability x_j=0



        u_j = self.prior.alphas[self.k_samples_prior, j]
        u_k = self.prior.alphas[self.k_samples_prior, j+1:]
        assert (u_j.shape == (self.N,) and u_k.shape == (self.N, self.J - (j+1)))

        x_j = self.x[:, j]
        x_i = self.x[:, :j]
        I = x_i.sum(axis=1)

        legal = np.logical_and(u_j > 10**-7, u_k.sum(axis=1) > 10**-7) # we aren't going to have a legal pdf where u_j or Σu_k equals zero

        unifs = np.empty(self.N,)
        unifs[np.logical_not(legal) ] = np.random.uniform(0, 1, size=(self.N - legal.sum(),))
        # unifs[legal] = betas[legal].cdf(x_j[legal] / (1-I)[legal])
        unifs[legal] = scipy.stats.beta(u_j[legal], u_k.sum(axis=1)[legal]).sf(x_j[legal] / (1-I)[legal])

        return unifs

    def build_x_uniform(self):
        for j in range(1, self.J):
            self.x_uniforms[:, j] = self.x_to_uniforms(j)

    def build_x_post(self):
        for j in range(1, self.J):
            self.samples_post[:,j] = self.uniforms_to_x(j, self.x_uniforms[:,j])

    def uniforms_to_x(self, j, uniforms):
        '''
        x_j+ | x_i+ for all i<j
        find the inverse cdf
        use the uniforms to draw samples for x_j+
        '''
        assert uniforms.shape == (self.N,), f'Uniforms must have shape {self.N} not {uniforms.shape}'

        u_j = self.prior.alphas[self.k_samples_post, j]
        u_k = self.prior.alphas[self.k_samples_post, j+1:]
        assert (u_j.shape == (self.N,) and u_k.shape == (self.N, self.J - (j+1)))

        x_i = self.samples_post[:, :j]
        I = x_i.sum(axis=1) # how much of the space has already been used up

        # initialize
        x_j = np.empty((self.N,))

        #if u_j=0 then x_j must be zero
        no_prob = u_j < 10e-7
        x_j[no_prob] = 0

        # if Σu_k=0 we must take the remaining probability.
        last_prob = u_k.sum(axis=1)<10e-7
        x_j[last_prob] = (1 - I)[last_prob]

        # in the general case we use the uniform to draw (x_j)/(1-I) from
        # the inverse CDF of a Beta with parameters u_j and Σu_k
        legal = np.logical_and(u_j>10e-7 , u_k.sum(axis=1)>10e-7)
        x_j[legal] = scipy.stats.beta(u_j[legal], u_k[legal].sum(axis=1)).isf(uniforms[legal]) * (1-I)[legal]

        return x_j

    def class_to_uniform(self, probs, ind):
        '''
        Given a probability vector describing several classes e.g. [0.1, 0.5, 0.4]
        and a class_index (e.g. 1)
        return a uniform from within the class probability interval (e.g. 0.1 - 0.6)
        '''
        stacked = probs.cumsum(axis=-1)
        stacked = np.column_stack((np.zeros_like(ind), stacked))
        lower = stacked[np.arange(self.prior.N), ind]
        upper = stacked[np.arange(self.prior.N), ind+1]
        draw = scipy.stats.uniform(loc=lower, scale=(upper - lower)).rvs()
        return draw

    def uniform_to_class(self, probs, unifs):
        # given a probability vector describing class mixing ratios
        # choose a class based on a uniform random variable
        # class indices:      0         1         2
        # class probs  :      0.3       0.3       0.4
        # stacked probs:  0        0.3       0.6       1.0
        #              :                         ^
        # uniform rv   :                         0.7
        # output class :  2
        probs /= probs.sum(axis=1, keepdims=1) # normalize just in case
        stacked = probs.cumsum(axis=-1)
        stacked = np.column_stack((np.zeros_like(unifs), stacked))
        # numpy's searchsorted doesn't work for arrays
        # so instead we column_stack unifs and probs into a single array
        # and define a function that searches a row (probs) for the first
        # element in the row (unif)
        # we can then apply this function along axis=1
        locate_first = lambda row: np.searchsorted(row[1:], row[0])
        # subtract 1 from the index because we inserted a column of zeroes at the beginning of stacked
        return np.apply_along_axis(locate_first, axis=1, arr=np.column_stack((unifs[:,None], stacked))) - 1
