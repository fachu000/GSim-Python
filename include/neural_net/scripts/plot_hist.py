# This script is to plot the training history in an interactive window.
#
# It is used to view the figures while running VS code on a server.
#

# %%
import importlib
from gsim.include.neural_net import neural_net

importlib.reload(neural_net)
from gsim.include.neural_net.neural_net import NeuralNet

# Complete with the desired folder
nn_folder = 'model_data/trained_models/transformer_4_madrid-all-mape'

hist = NeuralNet.load_hist_from_folder(nn_folder)
G = NeuralNet.plot_training_history(hist, first_step_to_plot=0)[0]
print(G.l_subplots[1].xlabel)
G.plot()

# %%
