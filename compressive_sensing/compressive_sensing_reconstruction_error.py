import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from cvxpy import *
import time

n=1000 
sparsity_list = np.linspace(0.5, 0.95, 60)
undersampling_list = np.linspace(1, 0.1, 60)
error_lasso = np.zeros((len(sparsity_list), len(undersampling_list)))
error_cvx = np.zeros((len(sparsity_list), len(undersampling_list)))
for (i_s, sparsity) in zip(range(len(sparsity_list)), sparsity_list):
    for (i_u, undersampling) in zip(range(len(undersampling_list)), undersampling_list):
        m=n*undersampling
        percent_zero=sparsity
        signal = 1.0*(np.random.rand(1, n) > percent_zero)
        signal = signal + noise_level * np.random.randn(1, n)
        sampling_scheme = 1.0*(np.random.randn(m, n))
        
        samples = np.dot(sampling_scheme, signal.T)
        
        gamma = Parameter(sign="positive")
        gamma.value=0.0001
        x = Variable(n)
        error = sum_squares(sampling_scheme*x - samples)
        obj = Minimize(error + gamma*norm(x, 1))
        #constraints = [norm(x, 1)<20]
        prob = Problem(obj)
        t = time.time()        
        #result = prob.solve(solver=ECOS)
        elapsed = time.time() - t
        clf = linear_model.Lasso(alpha=0.001, fit_intercept=False, positive=True)
        clf.fit(sampling_scheme, samples)
        delta = (signal-clf.coef_)
        error = np.sqrt(np.mean(delta * delta))
        error_lasso[i_s, i_u] = error
        #delta = (signal.T-np.array(x.value))
        #error = np.sqrt(np.mean(delta * delta))
        #error_cvx[i_s, i_u] = error
        print("%f/%f/%f/%f" % (sparsity, undersampling, error, elapsed))

plt.subplot(211)
plt.title("Results for cvx")

#plt.xticks(sparsity_list)
#plt.imshow(error_cvx, interpolation="none")
#plt.colorbar()
#plt.gca().set_yticks(sparsity_list)


plt.subplot(111)
plt.title("RMSE for reconstruction using lasso")
plt.imshow(error_lasso, interpolation="none", aspect='auto', extent=[1/undersampling_list[0],1/undersampling_list[-1],sparsity_list[-1],sparsity_list[0]])
plt.colorbar()
plt.xlabel('undersampling factor')
plt.ylabel('sparsity in percent')
#plt.axes().set_aspect('equal', 'datalim')