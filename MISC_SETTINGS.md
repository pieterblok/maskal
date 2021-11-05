The following settings can probably stay unchanged: <br/> <br/>

| Setting        			| Description        													|
| --------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| transfer_learning_on_previous_models	| Whether to use the weight-files of the previous trainings for transfer-learning					|
| warmup_iterations			| The number of warmup-iterations that can be used to stabilize the training process 			 		|
| train_iterations_base			| The number of training iterations to start the training with (this number of training iterations is used when the total number of training images is below the value of **step_image_number**)								 			 		|
| train_iterations_step_size		| When the number of training images exceeds the **step_image_number**, then this number of iterations is added to the **train_iterations_base**																	|
| step_image_number			| The number of training images to increase the number of iterations specified in **train_iterations_step_size**	|
| step_ratios				| When the training iterations reach this iteration ratio, then the learning rate is automatically lowered by a fraction of 0.1 |
| eval_period				| The number of training iterations when to do the evaluation on the validation set					|
| checkpoint_period			| The number of training iterations at which the weights are stored (use **-1** to disable intermediate checkpoints)	|
| weight_decay	 			| The weight-decay value to train Mask R-CNN										|
| learning_policy 			| The learning-policy to train Mask R-CNN										|
| gamma		 			| The gamma-value to train Mask R-CNN											|
| train_batch_size 			| The image batch-size that is used to train Mask R-CNN									|
| num_workers	 			| The number of workers to train Mask R-CNN										|
| train_sampler	 			| The data-sampler to train Mask R-CNN. Use **"RepeatFactorTrainingSampler"**, when there is class-imbalance		|
| minority_classes 			| Only when the **"RepeatFactorTrainingSampler"** is used: specify the minority-classes that have to be repeated	|
| repeat_factor_smallest_class		| Only when the **"RepeatFactorTrainingSampler"** is used: specify the repeat-factor of the smallest class (use a value higher than 1.0 to repeat the minority classes)																	|
| experiment_name			| Specify the name of your experiment											|
| strategies				| Use **'uncertainty'** to select the most uncertain images for the active learning. Other options are **'random'** and **'certainty'** |
| mode					| Uncertainty sampling method. Use **'mean'** when you want to sample the most uncertain images, use **'min'** when you want to sample the most uncertain instances																	|
| equal_pool_size			| When **True** this will sample the same **pool_size** for every sampling iteration. When **False**, an unequal **pool_size** will be sampled for the specified number of loops															|
| dropout_probability			| Specify the dropout probability between 0.1 and 0.9. Our experiments indicated that **0.25** is a good value		|
| mcd_iterations			| The number of Monte-Carlo iterations to calculate the uncertainty of the image. When this number is increased, the uncertainty metric will be more consistent. When this number is decreased, the sampling will be faster. The value **20** is a good compromise between consistency and speed	|
| dropout_method			| Use **'head'** to significantly speed up the sampling process. Use **'complete'** to do the image inference through the entire network (not recommended) 																		|
| al_batch_size				| Only when the **'complete'** option is used for the **dropout_method**. Image batch-size to do the uncertainty calculation|
| iou_thres				| Intersection of Union threshold to cluster the different instance segmentations into observations for the uncertainty calculation																			|
<br/>
