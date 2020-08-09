import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from matplotlib.pyplot import imshow
from PIL import Image

def testSOM(sourceSom, targetSom, sourceSomN, targetSomN, net, N_test, sourceData, targetData):
  mses, outputs = [], []
  im = False
  if len(targetData[0])>10:
    im=True
    outputs = np.array([])
  for i in range(N_test):
    activation_matrix = sourceSom.return_activation_matrix(sourceData[i])
    bmu_value = sourceSom.return_BMU_weights(sourceData[i])

    # Backward pass
    net.set_node_activations(sourceSomN, activation_matrix)
    mod_hebbian_matrix = net.compute_node_activations(targetSomN, set_node_matrix=False)
    max_row, max_col = np.unravel_index(mod_hebbian_matrix.argmax(), mod_hebbian_matrix.shape)
    mod_bmu_weights = targetSom.get_unit_weights(max_row, max_col)
    
    if im:
      mod_bmu_weights = np.array(mod_bmu_weights)
      mod_bmu_weights = mod_bmu_weights.astype(np.uint8)
      outputs = np.append(outputs, mod_bmu_weights)
    else:
      outputs.append(mod_bmu_weights)
    mse = mean_squared_error([targetData[i]], [mod_bmu_weights])
    mses.append(mse)
  if im:
    outputs = outputs.reshape(N_test, len(mod_bmu_weights))
  mses = np.array(mses)
  return outputs, np.mean(mses), np.std(mses)

def plotJoints(reals, outs):
  outputs = np.array(outs)
  real_SP, real_SR, real_ER = reals[:,0], reals[:,1], reals[:,2]
  recon_SP, recon_SR, recon_ER = outputs[:,0], outputs[:,1], outputs[:,2]
  plt.plot(real_SP, 'y', label="Actual SP")
  plt.plot(recon_SP, color='coral', label="Reconstructed SP")
  plt.plot(real_SR, 'm', label="Actual SR")
  plt.plot(recon_SR, color='violet', label="Reconstructed SR")
  plt.plot(real_ER, 'b', label="Actual ER")
  plt.plot(recon_ER, color='cyan', label="Reconstructed ER")
  plt.legend()
  plt.show()

def plotVisual(reals, outs, sampleNumber):
  outputs = np.array(outs)
  outputs = outputs.astype(np.uint8)

  im_1 = Image.fromarray(reals[sampleNumber].reshape(64, 64))
  im_2 = Image.fromarray(outputs[sampleNumber].reshape(64, 64))

  f = plt.figure()
  f.add_subplot(1,2, 1)
  plt.imshow(im_1, cmap='gray')
  f.add_subplot(1,2, 2)
  plt.imshow(im_2, cmap='gray')
  plt.show(block=True)

def testSomHubPass(sourceSom, hubSom, targetSom, sourceSomN, hubSomN, targetSomN, net, N_test, sourceData, targetData):
    mses, outs = [], []
    im = False
    if len(targetData[0])>10:
      im =True
      outs = np.array([])
    for i in range(N_test):
        mod0_activation_matrix = sourceSom.return_activation_matrix(sourceData[i])
        top_bmu_value = sourceSom.return_BMU_weights(sourceData[i])

        # First backward pass
        net.set_node_activations(sourceSomN, mod0_activation_matrix)
        mod1_hebbian_matrix = net.compute_node_activations(hubSomN, set_node_matrix=False)
        max_row, max_col = np.unravel_index(mod1_hebbian_matrix.argmax(), mod1_hebbian_matrix.shape)
        mod1_bmu_weights = hubSom.get_unit_weights(max_row, max_col)
        mod1_activation_matrix = hubSom.return_activation_matrix(mod1_bmu_weights)

        # Second backward pass
        net.set_node_activations(hubSomN, mod1_activation_matrix)
        mod2_hebbian_matrix = net.compute_node_activations(targetSomN, set_node_matrix=False)
        max_row, max_col = np.unravel_index(mod2_hebbian_matrix.argmax(), mod2_hebbian_matrix.shape)
        mod2_bmu_weights = targetSom.get_unit_weights(max_row, max_col)

        if im:
          mod2_bmu_weights = np.array(mod2_bmu_weights)
          mod2_bmu_weights = mod2_bmu_weights.astype(np.uint8)

          outs = np.append(outs, mod2_bmu_weights)
        else:
          outs.append(mod2_bmu_weights)
        # Compute MSE
        mse = mean_squared_error([targetData[i]], [mod2_bmu_weights])
        # print(mse)
        mses.append(mse)
    if im:
      outs = outs.reshape(N_test, len(mod2_bmu_weights))
    mses = np.array(mses)
    return outs, np.mean(mses), np.std(mses)

def get_SDH(som, data, n=50):      
    sdh = np.zeros((som._matrix_size, som._matrix_size))
    for sample in data:
        output_matrix = np.zeros((som._matrix_size, som._matrix_size))
        it = np.nditer(output_matrix, flags=['multi_index'])
        while not it.finished:
            # print "%d <%s>" % (it[0], it.multi_index),
            dist = som.return_euclidean_distance(sample, som._weights_matrix[it.multi_index[0], it.multi_index[1], :])
            output_matrix[it.multi_index[0], it.multi_index[1]] = dist
            it.iternext()

        output_matrix = output_matrix.flatten('F')
        idx = np.argpartition(output_matrix, n)
        for i in range(n):
            index = idx[i]
            x = int(index / som._matrix_size)
            y = int(index % som._matrix_size)
            sdh[x, y] = sdh[x, y] + 1
    levels = MaxNLocator(nbins=15).tick_values(sdh.min(), sdh.max())
    cmap = plt.get_cmap('YlOrRd')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    return sdh, cmap, norm

def get_activation_histogram(som, data_point):
    '''
    Activation histogram: distance of all units to the input data point
    '''
    act_h = som.return_normalized_distance_matrix(data_point)
    levels = MaxNLocator(nbins=15).tick_values(act_h.min(), act_h.max())
    cmap = plt.get_cmap('YlOrRd')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    return act_h, cmap, norm

def plotSom(reals, sampleNumber, som, image=True):

  data = reals[sampleNumber]
  if image:
    im_1 = Image.fromarray(data.reshape(64, 64))
  im_2, cmap_2, norm_2 = get_activation_histogram(som, data)
  im_3, cmap_3, norm_3 = get_SDH(som, data, n=4)
  
  
  f = plt.figure()
  
  
  f.add_subplot(1,3, 1)
  plt.imshow(im_1, cmap='gray')
  f.add_subplot(1,3, 2)
  plt.imshow(im_2, cmap=cmap_2, norm=norm_2)
  f.add_subplot(1,3, 3)
  plt.imshow(im_3, cmap=cmap_3, norm=norm_3)
  plt.show(block=True)
