# Change Log and To-do List
## Change Log
### 2020.02.11
1. Add an NgBoost demo.
2. Add DeepSpeed as the primary training framework. Since it seems that it speed up the GPU training more so than TPU, 
this will be the default optimizer framework. 
3. Start to work on Direct Policy Gradient.
4. Add a TF 1.13 version of BERT-class models. Here we have changed several things, such as 
adding label smoothing, and many things. Note that some of the features will be extremely difficult to implement
in TensorFlow. Therefore to implement this library it is only meant to start a feasible starting point.  
4. Integrated Hugging face transformers. We will start to add similar functionalities. 
5. The PyTorch version of Ranger is implemented. 

### 2020.02.12
1. Add a variable selector.
2. Finished key value pairs. High order group by.
3. Added discretizer.
4. Finished category encoder.
5. Finished target mean encoder.
6. Started to work on MetaDes and Direct Policy Gradient update. 

### 2020.02.13
1. Added Category Encoder
2. Added Discretizer. 
3. Added encoder for continuous variables. Needed to add icdf. 
4. Added some dimension reduction techniques. 
5. Added some codes for general model training frameworks. Not yet finished. 

##TO-DO
1. Need to start try to integrate DeepSpeed into the training frameworks by adding more optimizers.
2. For Tensorflow version, more models (some model will be coming very soon) need to be added.
For example T5 should be added. Furthermore, we will extract the final pooling layer and reverse order, 
therefore making it more prepared to work with models on top of language models. 
3. UDA must be implemented (Tensorflow and PyTorch)
4. Ranger must be implemented in TensorFlow. However, it seems that 
existing implementations are causing bugs, therefore, a rewrite from scratch is needed.
5. There will be two training frameworks for PyTorch. One is the one used by DeepSpeed, one is
the one used by the original hugging face implementations, but with TPU support. 
6. Some minor details should be added for the TF1 BERT version to make it more robust.
