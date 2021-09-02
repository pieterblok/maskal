# maskAL - active learning for Mask R-CNN in Detectron2

## Summary
Active learning automatically selects the most-informative images to annotate and retrain the algorithm. Active learning strives to reduce the number of annotations, without affecting the performance of the algorithm. Generally speaking, active learning involves the following steps:
1. Train an algorithm on a small subset of a bigger dataset
2. Infer the trained algorithm on the remaining dataset, and select the most-informative images with a sampling algorithm
3. Annotate the most-informative images, and then retrain the algorithm on the most informative-images
4. Repeat step 1-3 until the desired performance is reached (or when you are tired of doing annotations) <br/><br/>

![maskAL_graph](./demo/maskAL_vs_random.png?raw=true)

## maskAL installation
See [INSTALL.md](INSTALL.md)
<br/> <br/>

## Annotation procedure
See [ANNOTATION.md](../ANNOTATION.md)
<br/> <br/>

## Iterative sampling and annotation
We advise you to split the images and annotations in a training set, validation set and a test set. Remember that it is not required to annotate every single image in the folder, because we only want to annotate the most-informative images. <br/> 

1. From the big training set, a smaller initial set is randomly sampled (its size can be specified in the **maskAL.yaml** file). The images that do not have an annotation are placed in the **annotate** subfolder inside your image folder. You first need to annotate these images with labelme (json), v7-darwin (json) or cvat (xml). 
2. This procedure is repeated for the validation set and the test set (the file locations can be specified in the **maskAL.yaml** file). 
3. After the first training iteration, the sampling algorithm selects the most-informative images (its size can be specified as well in the **maskAL.yaml** file).
4. The most-informative images that don't have an annotation, are placed in the **annotate** subfolder, so that they can be annotated with labelme (json), v7-darwin (json) or cvat (xml). 
5. Step 4-5 are repeated for several training iterations. The number of iterations is specified in the **maskAL.yaml** file. 

Please note that this method does not work with the default COCO json-files of detectron2. These json-files summarize all annotations that have been completed before the training starts. Because active learning involves an iterative train and annotation procedure, these COCO-jsons lack the desired format.
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
<br/>

Description of the other settings in the maskAL.yaml file: [MISC_SETTINGS.md](../MISC_SETTINGS.md)
<br/>

Please refer to the folder **active_learning/config** for more setting-files. 
<br/> <br/>

## Citation
Please refer to our research article for more information or cross-referencing: 
<br/> <br/>

## License
Our software was forked from Detectron2 (https://github.com/facebookresearch/detectron2). As such, the software will be released under the [Apache 2.0 license](LICENSE). <br/><br/>

## Acknowledgements
maskAL was developed by Pieter Blok (pieter.blok@wur.nl).<br/><br/>
