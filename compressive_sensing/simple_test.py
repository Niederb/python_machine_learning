import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

n=1000 
m=200
percent_zero=0.98
signal = 1.0*(np.random.rand(1, n) > percent_zero)

sampling_scheme = 1.0*(np.random.randn(m, n))

samples = np.dot(sampling_scheme, signal.T)

clf = linear_model.Lasso(alpha=0.001, fit_intercept=False, positive=True)
clf.fit(sampling_scheme, samples)
print("Non zeros original signal: %i" % sum(signal.T > 0))
print("Non zeros reconstructed signal: %i" % sum(clf.coef_ > 0))

plt.subplot(311)
plt.plot(signal.T)
plt.xlabel("Original signal")
plt.subplot(312)
plt.plot(clf.coef_)
plt.xlabel("Reconstructed signal")
plt.subplot(313)
plt.plot((signal-clf.coef_).T)
plt.xlabel("Reconstructed error")

print("Reconstrudtion RMSE: %f" % np.sqrt(np.mean(delta * delta)))