
This document describes NeuralNet, which is a wrapper around torch.nn.Module. It
provides a framework for training, evaluation, and performing inference with
neural networks. 

# Data pipelines

Casting a given data pipeline into the NeuralNet architecture allows rapid
experimentation without having to write and debug a large amount of boilerplate
code, such as training loops, normalization, training history tracking, saving
and loading, implementing telemetry, etc.

By default, the framework assumes supervised learning, where the datasets
comprise input-target pairs. To accommodate self-supervised and unsupervised
learning, one can use the flag `no_targets`. For unsupervised learning, one
needs to override `_get_loss`, since the default one splits the input data into
input and target.  

To specify different steps of the data pipeline, one can use `DataAdapter`s and
`Normalizer`s. Examples of operations that a `DataAdapter` can perform are:
- Extracting features
- Resizing an input image
- Forming input-target pairs for self-supervised learning
- Mapping an output class to a string. 
- Applying transformations. 

A useful functionality of NeuralNet is that it allows applying the `DataAdapter`
offline via `load_or_create_preprocessed_dataset`. Saving the adapted dataset to disk
allows for faster experimentation. 

The pipeline at inference time (`NeuralNet.predict`) comprises the following
elements, organized into layers:


        raw input                                                             raw output
            |                                                                      |
            v                                                                      |
|------------------------------|                                    |------------------------------|
|    DataAdapter.adapt_input   |                                    |   DataAdapter.adapt_output   |
|------------------------------|                                    |------------------------------|
            |                                                                      ^
            | unnormalized input (typ. features)                                   | unnormalized output
            v                                                                      |
|----------------------------------|                              |----------------------------------|
|   batch formation (collate_fn)   |                              |           uncollate_fn           |
|----------------------------------|                              |----------------------------------|
            |                                                                      ^
            | unnormalized batches                                                 | unnormalized output batches
            v                                                                      |
|----------------------------------|                              |-------------------------------------|
| Normalizer.normalize_input_batch |                              | Normalizer.unnormalize_output_batch |
|----------------------------------|                              |-------------------------------------|
            | normalized batches                                                   ^
            |                          |----------------------|                    |
            |------------------------->|  NeuralNet.forward() |--------------------|
                                       |----------------------|       normalized output batches
                                                

The pipeline for supervised training (`NeuralNet.fit`) and evaluation
(`NeuralNet.evaluate`) is as follows:

                                             dataset item
    ^                                              |
    | precomputable               |--------------------------------|                                   
----|------                       | DataAdapter.adapt_dataset_item | (if no_targets=False) 
                                  |--------------------------------|                                                   
                                                   |  input-target pairs
                                  |--------------------------------|                                   
                                  |  batch formation (collate_fn)  |  
                                  |--------------------------------|                                                   
                                      |                     |
                                      |                     |                                                             
            |-------------------------|                     |----------------------|
            | unnormalized input batches                                           | unnormalized targets
            v                                                                      v
|----------------------------------|                              |------------------------------------|
| Normalizer.normalize_input_batch |                              | Normalizer.normalize_targets_batch |
|----------------------------------|                              |------------------------------------|
            |                                                                      |
            | normalized input batches                                             | normalized target batches
            v                                                                      |
|----------------------------------|                                               |
|     NeuralNet.forward()          |                                               |
|----------------------------------|                                               | 
            | normalized output batches                                            |
            |                          |----------------------|                    |
            |------------------------->|         loss         |<-------------------|
                                       |----------------------|       
                                                  |

Note that unnormalized losses can also be computed; cf. `normalizers.py`. 

If `no_targets=True`, then `DataAdapter.adapt_dataset_item` is replaced with
`DataAdapter.adapt_input`. 

By default, `DataAdapter.adapt_dataset_item` will split the input item into
input and target. It will then apply `DataAdapter.adapt_input` and
`DataAdapter.adapt_target` to the respective parts. This is done in this way to
accommodate cases where both inputs and targets require need to be adapted
together. 

The methods `adapt_dataset_item`, `adapt_input`, and `adapt_target` of
`DataAdapter` can be the result of the composition of several operations:
- one that can be precomputed and saved (`load_or_create_preprocessed_dataset`)
- one that needs to be performed at loss computation time (training and
  evaluation). For example, forming input-target pairs for self-supervised
  learning. 
- one that needs to be always performed, such as feature extraction. 

For this reason, these methods are given the following an object of class
`AdaptationSpec`, with the following properties:
- `preprocess_only` (bool): This will be true iff the method is called from
  `load_or_create_preprocessed_dataset`. 
- `input_already_preprocessed` (bool): This will be true iff the method is called on
  data for which the method has already been called with `preprocess_only=True`.
  This will happen when the method is called on a dataset created by
  `load_or_create_preprocessed_dataset`, saved, and then read from disk. 
- `inference` (bool): This will be true if the method is called from
  `NeuralNet.predict` and false if it is called from `fit`, `evaluate`, or
  `load_or_create_preprocessed_dataset`. In self-supervised learning, this flag
  can be used to know whether to form input-target pairs or not.

# Special cases

## Self-supervised learning

In self-supervised learning, one would like to create input-target pairs on the
fly from the raw input data. By "on the fly" we mean that the input-target pairs
are not precomputed and saved to disk, but rather generated training/evaluation
time. To this end, the best practice is as follows:

Create a `DataAdapter` whose `adapt_input` method returns input-target pairs
whenever `AdaptationSpec.preprocess_only=False` and
`AdaptationSpec.inference=False`. To ensure that the flag `no_targets` is set
correctly downstream, override `DataAdapter.get_no_targets` to return False in
this case as well. 






