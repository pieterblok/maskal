# maskAL - Active learning for Mask R-CNN in Detectron2

<p align="center">
  <img src="./demo/maskAL_framework.png?raw=true" alt="maskAL_framework"/>
</p>

## Summary
Active learning automatically selects the most-informative images to annotate and retrain the algorithm. By using active learning, we aim to reduce the number of annotations, without affecting the performance of the algorithm. Generally speaking, active learning involves the following steps:
1. Train an algorithm on a small subset of a bigger dataset
2. Use the trained algorithm to make predictions on the remaining dataset, and select the most-informative images with a sampling algorithm
3. Annotate the most-informative images, and then retrain the algorithm on the most informative-images
4. Repeat step 1-3 until the desired performance is reached (or when you are tired of doing annotations) <br/><br/>

The figure below shows the potential of active learning on our dataset. The active learning reached a similar performance after sampling 1100 images as the random image sampling after 2500 images, indicating that 1400 annotations could have been saved (see the black dashed line):

![maskAL_graph](./demo/maskAL_vs_random.png?raw=true)

## maskAL installation
See [INSTALL.md](INSTALL.md)
<br/> <br/>

## Data preparation
We advise you to split the images and annotations in a training set, validation set and a test set. Remember that it is not required to annotate every single image in the folder, because we only want to annotate the most-informative images. <br/> 

1. From the big training set, a smaller initial set is randomly sampled (its size can be specified in the **maskAL.yaml** file). The images that do not have an annotation are placed in the **annotate** subfolder inside your image folder. You first need to annotate these images with LabelMe (json), V7-Darwin (json) or CVAT (xml) (when using CVAT, export the annotations to **LabelMe 3.0** format). Refer to our annotation procedure: [ANNOTATION.md](ANNOTATION.md) 
2. This procedure is repeated for the validation set and the test set (the file locations can be specified in the **maskAL.yaml** file). 
3. After the first training iteration, the sampling algorithm selects the most-informative images (its size can be specified as well in the **maskAL.yaml** file).
4. The most-informative images that don't have an annotation, are placed in the **annotate** subfolder, so that they can be annotated with LabelMe (json), V7-Darwin (json) or CVAT (xml) (when using CVAT, export the annotations to **LabelMe 3.0** format). 
5. You can also use the trained model to make predictions on the unlabelled images to further reduce annotation time. Activate **auto_annotate** in the **maskAL.yaml** file, and specify the **export_format** (currently only **'labelme'** and **'cvat'** are supported). 
6. Step 3-5 are repeated for several training iterations. The number of iterations (**loops**) is specified in the **maskAL.yaml** file.

Please note that this method does not work with the default COCO json-files of detectron2. These json-files contain all annotations that have been completed before the training starts. Because active learning involves an iterative train and annotation procedure, these COCO-jsons lack the desired format.
<br/><br/>

## How to use maskAL
1. open a terminal
2. cd maskAL
3. activate the maskAL virtual environment (conda activate maskAL)
4. python maskAL.py --config maskAL.yaml <br/> <br/>

Change the following settings in the maskAL.yaml file: <br/>

| Setting        	| Description           														|
| ----------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| weightsroot	        | The file directory where the weight-files are stored											|
| resultsroot		| The file directory where the result-files are stored 											|
| dataroot	 	| The root directory where all image-files are stored											|
| traindir	 	| The file directory where the training images and annotations are stored								|
| valdir	 	| The file directory where the validation images and annotations are stored								|
| testdir	 	| The file directory where the test images and annotations are stored									|
| cuda_visible_devices 	| The identifiers of the CUDA device(s) you want to use for training and sampling							|
| classes	 	| The names of the classes of the annotated instances											|
| learning_rate	 	| The learning-rate to train Mask R-CNN (default value: 0.01)										|
| confidence_threshold 	| Confidence-threshold for the image inference with Mask R-CNN (default value: 0.5)							|
| nms_threshold 	| Non-maximum suppression threshold for the image inference with Mask R-CNN (default value: 0.3)					|
| initial_datasize 	| The size of the initial dataset, which will be randomly pooled from the traindir when starting the active learning (default value: 100)|
| pool_size	 	| The number of selected images from the traindir when doing the active learning sampling (default value: 200)				|
| loops		 	| The number of active learning sampling iterations (default value: 5)									|
| auto_annotate	 	| Set this to **True** when you want the model to make predictions on the unlabelled images to further reduce annotation time 		|
| export_format	 	| When auto_annotate is activated: specifiy the export-format of the annotations (currently supported formats: **'labelme'**, **'cvat'**)	|
<br/>

Description of the other settings in the maskAL.yaml file: [MISC_SETTINGS.md](MISC_SETTINGS.md)
<br/>

Please refer to the folder **active_learning/config** for more setting-files. 
<br/> <br/>

## Troubleshooting
See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
<br/> <br/>

## Citation
Please refer to our research article for more information or cross-referencing: 
<br/> <br/>

## License
Our software was forked from Detectron2 (https://github.com/facebookresearch/detectron2). As such, the software will be released under the [Apache 2.0 license](LICENSE). <br/> <br/>

## Acknowledgments
Two of our software methods are inspired by RovelMan's software: <br/>
https://github.com/RovelMan/active-learning-framework<br/> <br/>

## Contact
maskAL is developed and maintained by Pieter Blok. <br/> <br/>
