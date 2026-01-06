import numpy as np
import scipy.io

srate = 512
X = np.random.randn(8, 1000)  # 8 channels, 1000 samples

# Save to a .mat file
scipy.io.savemat('./data/test_data.mat', {'X': X})