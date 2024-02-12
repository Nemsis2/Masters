import torch as th
import torch.nn.utils.prune as prune
from helper_scripts import *


#https://github.com/sahandilshan/Simple-NN-Compression/blob/main/Simple_MNIST_Compression.ipynb
def get_pruned_parameters_countget_pruned_parameters_count(pruned_model):
    params = 0
    for param in pruned_model.parameters():
        if param is not None:
            params += th.nonzero(param).size(0)
    return params

"""
perform pruning on a resnet model
"""
def prune_model(model, percentage):
      total_params_before = get_pruned_parameters_countget_pruned_parameters_count(model)
      l = nested_children(model)

      parameters_to_prune = []
      parameters = []
      for outer_key in l.keys():
            if "layer" in outer_key:
                  for inner_key in l[outer_key]:
                        for nets in l[outer_key][inner_key].keys():
                              if ("conv" or "bn") in nets:
                                    parameters_to_prune.append((l[outer_key][inner_key][nets], 'weight'))
                                    parameters.append(l[outer_key][inner_key][nets])

      prune.global_unstructured(
      parameters_to_prune,
      pruning_method=prune.L1Unstructured,
      amount= percentage, # Specifying the percentage
      )

      for layer in parameters:
            prune.remove(layer, 'weight')

      total_params_after = get_pruned_parameters_countget_pruned_parameters_count(model)

      return model, (total_params_after/total_params_before)
