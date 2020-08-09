import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pyERA.pyERA.som import Som
from pyERA.pyERA.utils import ExponentialDecay
from pyERA.pyERA.hebbian import HebbianNetwork


########################### Data Preprocessing ##################################

filePtrain = ".\\Nao\\database\\preprocessedDataTrainS.npy"
filePtest = ".\\Nao\\database\\preprocessedDataTest.npy"
fileNtrain = ".\\Nao\\database\\normalizedDataTrainS.npy"
fileNtest = ".\\Nao\\database\\normalizedDataTest.npy"

sampleLen = 8196
N_test = 100
N_train = 2000
raw_train = np.load(fileNtrain)
raw_train = raw_train.reshape(N_train, sampleLen)
train_joints = raw_train[:, 1:4].astype(float)
train_topCam = raw_train[:, 4:4100].astype(np.uint8)
train_botCam = raw_train[:, 4100:].astype(np.uint8)

raw_test = np.load(fileNtest)
raw_test = raw_test.reshape(N_test, sampleLen)
test_joints = raw_test[:, 1:4].astype(float)
test_topCam = raw_test[:, 4:4100].astype(np.uint8)
test_botCam = raw_test[:, 4100:].astype(np.uint8)

tot_sample = N_train


##################### Testversion: Connect Proprioceptive SOM and SOM of bottom Camera directly ########################################
'''
som_size = 30
visual_som = Som(matrix_size=som_size, input_size= len(train_botCam[0]))
proprio_som = Som(matrix_size=som_size, input_size= len(train_joints[1]))

hebbian_network = HebbianNetwork("net")
hebbian_network.add_node("visual_som", (som_size, som_size))
hebbian_network.add_node("proprio_som", (som_size, som_size))
hebbian_network.add_connection(0, 1)

hebbian_network.print_info()

################ Training ########################################################
tot_epoch = 1
my_learning_rate = ExponentialDecay(starter_value=0.5, decay_step=tot_epoch * tot_sample/ 5, decay_rate=0.9, staircase=True)
my_radius = ExponentialDecay(starter_value=np.rint(som_size / 3), decay_step=tot_epoch * tot_sample/ 6, decay_rate=0.90,
							 staircase=True)

tot_sample =100
print("Training starts.")
for epoch in range(1, tot_epoch+1):
	for sample in range(1, tot_sample):

		learning_rate = my_learning_rate.return_decayed_value(global_step=epoch * tot_sample + sample)
		radius = my_radius.return_decayed_value(global_step=epoch * tot_sample + sample)

		# Training the Visual SOM
		input_vector = train_botCam[sample]
		bmu_index = visual_som.return_BMU_index(input_vector)
		bmu_weights = visual_som.get_unit_weights(bmu_index[0], bmu_index[1])
		bmu_neighborhood_list = visual_som.return_unit_round_neighborhood(bmu_index[0], bmu_index[1], radius=radius)
		visual_activation_matrix = visual_som.return_activation_matrix(input_vector)
		visual_som.training_single_step(input_vector, units_list=bmu_neighborhood_list, learning_rate=learning_rate, radius=radius, weighted_distance=False)

		# Training the Proprioceptive SOM
		input_vector = train_joints[sample]
		bmu_index = proprio_som.return_BMU_index(input_vector)
		bmu_weights = proprio_som.get_unit_weights(bmu_index[0], bmu_index[1])
		bmu_neighborhood_list = proprio_som.return_unit_round_neighborhood(bmu_index[0], bmu_index[1], radius=radius)
		proprio_activation_matrix = proprio_som.return_activation_matrix(input_vector)
		proprio_som.training_single_step(input_vector, units_list=bmu_neighborhood_list, learning_rate=learning_rate, radius=radius, weighted_distance=False)

		# Hebbian Learning
		hebbian_network.set_node_activations(0, visual_activation_matrix)
		hebbian_network.set_node_activations(1, proprio_activation_matrix)
		hebbian_network.learning(learning_rate=0.1, rule="hebb")

print("Training is done.")

'''

############### Test the Testversion #######################################################

'''
# Recreate joint data from visual data
mses = []
for i in range(N_test):
	visual_activation_matrix = visual_som.return_activation_matrix(test_botCam[i])
	visual_bmu_value = visual_som.return_BMU_weights(test_botCam[i])

	# Backward pass
	hebbian_network.set_node_activations(0, visual_activation_matrix)
	proprio_hebbian_matrix = hebbian_network.compute_node_activations(1, set_node_matrix=False)
	max_row, max_col = np.unravel_index(proprio_hebbian_matrix.argmax(), proprio_hebbian_matrix.shape)
	proprio_bmu_weights = proprio_som.get_unit_weights(max_row, max_col)

	# print(test_joints[i])
	# print(proprio_bmu_weights)
	# Compute MSE
	mse = mean_squared_error([test_joints[i]], [proprio_bmu_weights])
	# print(mse)
	mses.append(mse)

mses = np.array(mses)
print("Predicting joint data from visual:")
print("mean MSE:")
print(np.mean(mses))
print("std:")
print(np.std(mses))

# Recreate visual data from joint data
mses = []
for i in range(N_test):
	proprio_activation_matrix = proprio_som.return_activation_matrix(test_joints[i])
	proprio_bmu_value = proprio_som.return_BMU_weights(test_joints[i])

	# Backward pass
	hebbian_network.set_node_activations(1, proprio_activation_matrix)
	visual_hebbian_matrix = hebbian_network.compute_node_activations(0, set_node_matrix=False)
	max_row, max_col = np.unravel_index(visual_hebbian_matrix.argmax(), visual_hebbian_matrix.shape)
	visual_bmu_weights = visual_som.get_unit_weights(max_row, max_col)

	# print(test_botCam[i])
	# print(visual_bmu_weights)
	# Compute MSE
	mse = mean_squared_error([test_botCam[i]], [visual_bmu_weights])
	# print(mse)
	mses.append(mse)

mses = np.array(mses)

print("Predicting visual data from joint:")
print("mean MSE:")
print(np.mean(mses))
print("std:")
print(np.std(mses))
'''

#################### 2. Version: Hub SOM ############################################


som_size = 30
top_visual_som = Som(matrix_size=som_size, input_size= len(train_topCam[0]))
bottom_visual_som = Som(matrix_size=som_size, input_size= len(train_botCam[0]))
proprio_som = Som(matrix_size=som_size, input_size= len(train_joints[1]))


hebbian_network = HebbianNetwork("net")
hebbian_network.add_node("top_visual_som", (som_size, som_size))
hebbian_network.add_node("bottom_visual_som", (som_size, som_size))
hebbian_network.add_node("proprio_som", (som_size, som_size))
hebbian_network.add_connection(0, 2)
hebbian_network.add_connection(1, 2)

hebbian_network.print_info()



##################### Training #############################################
tot_epoch = 1
my_learning_rate = ExponentialDecay(starter_value=0.5, decay_step=tot_epoch * tot_sample/ 5, decay_rate=0.9, staircase=True)
my_radius = ExponentialDecay(starter_value=np.rint(som_size / 3), decay_step=tot_epoch * tot_sample/ 6, decay_rate=0.90,
							 staircase=True)

print("Training starts.")
for epoch in range(1, tot_epoch+1):
	for sample in range(1, tot_sample):

		learning_rate = my_learning_rate.return_decayed_value(global_step=epoch * tot_sample + sample)
		radius = my_radius.return_decayed_value(global_step=epoch * tot_sample + sample)

		# Training the bottom visual SOM
		input_vector = train_botCam[sample]
		bmu_index = bottom_visual_som.return_BMU_index(input_vector)
		bmu_weights = bottom_visual_som.get_unit_weights(bmu_index[0], bmu_index[1])
		bmu_neighborhood_list = bottom_visual_som.return_unit_round_neighborhood(bmu_index[0], bmu_index[1], radius=radius)
		bottom_visual_activation_matrix = bottom_visual_som.return_activation_matrix(input_vector)
		bottom_visual_som.training_single_step(input_vector, units_list=bmu_neighborhood_list, learning_rate=learning_rate, radius=radius, weighted_distance=False)

		# Training the top visual SOM
		input_vector = train_topCam[sample]
		bmu_index = top_visual_som.return_BMU_index(input_vector)
		bmu_weights = top_visual_som.get_unit_weights(bmu_index[0], bmu_index[1])
		bmu_neighborhood_list = top_visual_som.return_unit_round_neighborhood(bmu_index[0], bmu_index[1],
																				 radius=radius)
		top_visual_activation_matrix = top_visual_som.return_activation_matrix(input_vector)
		top_visual_som.training_single_step(input_vector, units_list=bmu_neighborhood_list,
											   learning_rate=learning_rate, radius=radius, weighted_distance=False)

		# Training the Proprioceptive SOM
		input_vector = train_joints[sample]
		bmu_index = proprio_som.return_BMU_index(input_vector)
		bmu_weights = proprio_som.get_unit_weights(bmu_index[0], bmu_index[1])
		bmu_neighborhood_list = proprio_som.return_unit_round_neighborhood(bmu_index[0], bmu_index[1], radius=radius)
		proprio_activation_matrix = proprio_som.return_activation_matrix(input_vector)
		proprio_som.training_single_step(input_vector, units_list=bmu_neighborhood_list, learning_rate=learning_rate, radius=radius, weighted_distance=False)

		# Hebbian Learning
		hebbian_network.set_node_activations(0, top_visual_activation_matrix)
		hebbian_network.set_node_activations(1, bottom_visual_activation_matrix)
		hebbian_network.set_node_activations(2, proprio_activation_matrix)
		hebbian_network.learning(learning_rate=0.1, rule="hebb")

print("Training is done.")
top_visual_som.save(name="top_visual_som")
bottom_visual_som.save(name="bottom_visual_som")
proprio_som.save(name="proprio_som")
for connection in hebbian_network._connection_list:
	weights = connection["Connection"]._weights_matrix
	np.savez("./connection"+ str(connection["Start"]) + str(connection["End"]), weights)



#################### 3. Version: Explicit Hub SOM - Trained on node ############################################

som_size = 30
top_visual_som = Som(matrix_size=som_size, input_size= len(train_topCam[0]))
bottom_visual_som = Som(matrix_size=som_size, input_size= len(train_botCam[0]))
proprio_som = Som(matrix_size=som_size, input_size= len(train_joints[1]))
hub_som = Som(matrix_size=som_size, input_size=6)


hebbian_network = HebbianNetwork("net")
hebbian_network.add_node("hub_som", (som_size, som_size))
hebbian_network.add_node("top_visual_som", (som_size, som_size))
hebbian_network.add_node("bottom_visual_som", (som_size, som_size))
hebbian_network.add_node("proprio_som", (som_size, som_size))
hebbian_network.add_connection(1, 0)
hebbian_network.add_connection(2, 0)
hebbian_network.add_connection(3, 0)

hebbian_network.print_info()



##################### Training #############################################
tot_epoch = 1
my_learning_rate = ExponentialDecay(starter_value=0.5, decay_step=tot_epoch * tot_sample/ 5, decay_rate=0.9, staircase=True)
my_radius = ExponentialDecay(starter_value=np.rint(som_size / 3), decay_step=tot_epoch * tot_sample/ 6, decay_rate=0.90,
							 staircase=True)

print("Training starts.")
for epoch in range(1, tot_epoch+1):
	for sample in range(1, tot_sample):

		learning_rate = my_learning_rate.return_decayed_value(global_step=epoch * tot_sample + sample)
		radius = my_radius.return_decayed_value(global_step=epoch * tot_sample + sample)

		# Training the bottom visual SOM
		input_vector = train_botCam[sample]
		bv_bmu_index = bottom_visual_som.return_BMU_index(input_vector)
		bmu_weights = bottom_visual_som.get_unit_weights(bv_bmu_index[0], bv_bmu_index[1])
		bmu_neighborhood_list = bottom_visual_som.return_unit_round_neighborhood(bv_bmu_index[0], bv_bmu_index[1], radius=radius)
		bottom_visual_activation_matrix = bottom_visual_som.return_activation_matrix(input_vector)
		bottom_visual_som.training_single_step(input_vector, units_list=bmu_neighborhood_list, learning_rate=learning_rate, radius=radius, weighted_distance=False)

		# Training the top visual SOM
		input_vector = train_topCam[sample]
		tv_bmu_index = top_visual_som.return_BMU_index(input_vector)
		bmu_weights = top_visual_som.get_unit_weights(tv_bmu_index[0], tv_bmu_index[1])
		bmu_neighborhood_list = top_visual_som.return_unit_round_neighborhood(tv_bmu_index[0], tv_bmu_index[1],
																				 radius=radius)
		top_visual_activation_matrix = top_visual_som.return_activation_matrix(input_vector)
		top_visual_som.training_single_step(input_vector, units_list=bmu_neighborhood_list,
											   learning_rate=learning_rate, radius=radius, weighted_distance=False)

		# Training the Proprioceptive SOM
		input_vector = train_joints[sample]
		p_bmu_index = proprio_som.return_BMU_index(input_vector)
		bmu_weights = proprio_som.get_unit_weights(p_bmu_index[0], p_bmu_index[1])
		bmu_neighborhood_list = proprio_som.return_unit_round_neighborhood(p_bmu_index[0], p_bmu_index[1], radius=radius)
		proprio_activation_matrix = proprio_som.return_activation_matrix(input_vector)
		proprio_som.training_single_step(input_vector, units_list=bmu_neighborhood_list, learning_rate=learning_rate, radius=radius, weighted_distance=False)

		# Training the Hub SOM
		input_vector = np.array([tv_bmu_index[0], tv_bmu_index[1], bv_bmu_index[0], bv_bmu_index[1], p_bmu_index[0], p_bmu_index[1]])
		bmu_index = hub_som.return_BMU_index(input_vector)
		bmu_weights = hub_som.get_unit_weights(bmu_index[0], bmu_index[1])
		bmu_neighborhood_list = hub_som.return_unit_round_neighborhood(bmu_index[0], bmu_index[1], radius=radius)
		hub_activation_matrix = hub_som.return_activation_matrix(input_vector)
		hub_som.training_single_step(input_vector, units_list=bmu_neighborhood_list, learning_rate=learning_rate,
										 radius=radius, weighted_distance=False)

		# Hebbian Learning
		hebbian_network.set_node_activations(0, hub_activation_matrix)
		hebbian_network.set_node_activations(1, top_visual_activation_matrix)
		hebbian_network.set_node_activations(2, bottom_visual_activation_matrix)
		hebbian_network.set_node_activations(3, proprio_activation_matrix)
		hebbian_network.learning(learning_rate=0.1, rule="hebb")

print("Training is done.")
top_visual_som.save(name="top_visual_som")
bottom_visual_som.save(name="bottom_visual_som")
proprio_som.save(name="proprio_som")
hub_som.save(name="hub_som")
for connection in hebbian_network._connection_list:
	weights = connection["Connection"]._weights_matrix
	np.savez("./connection"+ str(connection["Start"]) + str(connection["End"]), weights)
	

#################### 4. Version: Explicit Hub SOM - Trained on concatenated data ############################################

som_size = 30
top_visual_som = Som(matrix_size=som_size, input_size= len(train_topCam[0]))
bottom_visual_som = Som(matrix_size=som_size, input_size= len(train_botCam[0]))
proprio_som = Som(matrix_size=som_size, input_size= len(train_joints[1]))
hub_som = Som(matrix_size=som_size, input_size=len(train_botCam[0])+len(train_topCam[0])+len(train_joints[1]))


hebbian_network = HebbianNetwork("net")
hebbian_network.add_node("hub_som", (som_size, som_size))
hebbian_network.add_node("top_visual_som", (som_size, som_size))
hebbian_network.add_node("bottom_visual_som", (som_size, som_size))
hebbian_network.add_node("proprio_som", (som_size, som_size))
hebbian_network.add_connection(1, 0)
hebbian_network.add_connection(2, 0)
hebbian_network.add_connection(3, 0)

hebbian_network.print_info()



##################### Training #############################################
tot_epoch = 1
my_learning_rate = ExponentialDecay(starter_value=0.5, decay_step=tot_epoch * tot_sample/ 5, decay_rate=0.9, staircase=True)
my_radius = ExponentialDecay(starter_value=np.rint(som_size / 3), decay_step=tot_epoch * tot_sample/ 6, decay_rate=0.90,
							 staircase=True)

print("Training starts.")
for epoch in range(1, tot_epoch+1):
	for sample in range(1, tot_sample):

		learning_rate = my_learning_rate.return_decayed_value(global_step=epoch * tot_sample + sample)
		radius = my_radius.return_decayed_value(global_step=epoch * tot_sample + sample)

		# Training the bottom visual SOM
		input_vector_1 = train_botCam[sample]
		bv_bmu_index = bottom_visual_som.return_BMU_index(input_vector_1)
		bmu_weights = bottom_visual_som.get_unit_weights(bv_bmu_index[0], bv_bmu_index[1])
		bmu_neighborhood_list = bottom_visual_som.return_unit_round_neighborhood(bv_bmu_index[0], bv_bmu_index[1], radius=radius)
		bottom_visual_activation_matrix = bottom_visual_som.return_activation_matrix(input_vector_1)
		bottom_visual_som.training_single_step(input_vector_1, units_list=bmu_neighborhood_list, learning_rate=learning_rate, radius=radius, weighted_distance=False)

		# Training the top visual SOM
		input_vector_2 = train_topCam[sample]
		tv_bmu_index = top_visual_som.return_BMU_index(input_vector_2)
		bmu_weights = top_visual_som.get_unit_weights(tv_bmu_index[0], tv_bmu_index[1])
		bmu_neighborhood_list = top_visual_som.return_unit_round_neighborhood(tv_bmu_index[0], tv_bmu_index[1],
																				 radius=radius)
		top_visual_activation_matrix = top_visual_som.return_activation_matrix(input_vector_2)
		top_visual_som.training_single_step(input_vector_2, units_list=bmu_neighborhood_list,
											   learning_rate=learning_rate, radius=radius, weighted_distance=False)

		# Training the Proprioceptive SOM
		input_vector_3 = train_joints[sample]
		p_bmu_index = proprio_som.return_BMU_index(input_vector_3)
		bmu_weights = proprio_som.get_unit_weights(p_bmu_index[0], p_bmu_index[1])
		bmu_neighborhood_list = proprio_som.return_unit_round_neighborhood(p_bmu_index[0], p_bmu_index[1], radius=radius)
		proprio_activation_matrix = proprio_som.return_activation_matrix(input_vector_3)
		proprio_som.training_single_step(input_vector_3, units_list=bmu_neighborhood_list, learning_rate=learning_rate, radius=radius, weighted_distance=False)

		# Training the Hub SOM
		input_vector = np.concatenate((input_vector_1, input_vector_2, input_vector_3))
		print(input_vector.shape)
		bmu_index = hub_som.return_BMU_index(input_vector)
		bmu_weights = hub_som.get_unit_weights(bmu_index[0], bmu_index[1])
		bmu_neighborhood_list = hub_som.return_unit_round_neighborhood(bmu_index[0], bmu_index[1], radius=radius)
		hub_activation_matrix = hub_som.return_activation_matrix(input_vector)
		hub_som.training_single_step(input_vector, units_list=bmu_neighborhood_list, learning_rate=learning_rate,
										 radius=radius, weighted_distance=False)

		# Hebbian Learning
		hebbian_network.set_node_activations(0, hub_activation_matrix)
		hebbian_network.set_node_activations(1, top_visual_activation_matrix)
		hebbian_network.set_node_activations(2, bottom_visual_activation_matrix)
		hebbian_network.set_node_activations(3, proprio_activation_matrix)
		hebbian_network.learning(learning_rate=0.1, rule="hebb")

print("Training is done.")
top_visual_som.save(name="top_visual_som")
bottom_visual_som.save(name="bottom_visual_som")
proprio_som.save(name="proprio_som")
hub_som.save(name="hub_som")
for connection in hebbian_network._connection_list:
	weights = connection["Connection"]._weights_matrix
	np.savez("./connection"+ str(connection["Start"]) + str(connection["End"]), weights)









