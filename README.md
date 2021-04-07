# maskAL - Active learning for Mask R-CNN

## Summary
Active learning automatically selects the most-informative images to annotate and retrain the algorithm. Active learning strives to reduce the number of annotations, without affecting the performance of the algorithm. Generally speaking, active learning involves the following steps:
1. Train an algorithm on a small subset of a bigger dataset
2. Infer the trained algorithm on the remaining dataset, and select the most-informative images with a sampling algorithm
3. Annotate the most-informative images, and then retrain the algorithm on the most informative-images
4. Repeat step 1-3 until the desired performance is reached (or we you are tired of doing annotations) <br/><br/>

## Installation of the software
See [INSTALL.md](../INSTALL.md)
<br/> <br/>

## Annotation procedure
Because active learning involves an iterative annotation of the most-informative images, we assume that the training, validation and test images have their corresponding json annotations from the **labelme** program. Please find the installation instructions of labelme on: https://github.com/wkentaro/labelme.
<br/>

**It is not required to annotate every single image, because the active learning algorithm will select the most-informative images for you to annotate.** <br/>

*Annotation procedure (labelme):*
1. Annotate each individual object by either a polygon, a circle or a rectangle. For our use-case, the polygon was the default shape.
2. Assign the correct class-name to the object. 
3. When an object consists of two or more separated parts: draw each separated part by an individual polygon and link the polygons by the same group id.
<br/> <br/> ![LabelMe annotation](./demo/labelme_annotation.png?raw=true)
<br/> *The broccoli head of the class "cateye" is occluded by a leaf, causing two separated instances of the broccoli head. Annotate the individual instances by a separate polygon and link them with an unique group_id (in this example 1). Suppose there is another occluded broccoli head in the image: then use another group_id (for example 2).*
<br/> <br/>

## Dataset preparation
Place all images and annotations in one folder. Remember that it is not required to annotate every single image in the folder, because we only want to annotate the most-informative images. <br/> 

1. The active-learning algorithm random samples a big train-set (the size of the train-set is specified by a ratio). 
2. From the big train-set, a smaller initial train-set is randomly sampled (its size can be specified). The images that do not have an annotation are placed in the **annotate subfolder** inside your image folder. You first need to annotate these images with labelme. 
3. The same procedure is repeated for the validation and test-set (the size of these sets are specified with another ratio). 
4. After the first training iteration, the sampling algorithm will infer the algorithm on the remaining images of the big train-set, to select the ones that are most-informative. The size of this  **image pool** can be specified as well.
5. The images in the pool that lack an annotation, are placed in the **annotate subfolder**, so that they can be annotated with labelme. 
6. Step 4-5 are repeated for several training iterations. The number of iterations can be specified. 

<br/>**Please note that this method does not work with the default COCO json-files of detectron2. These json-files summarize all annotations that have been completed before the training starts. Because active learning involves an iterative train and annotation procedure, these COCO-jsons lack the desired format.** 
<br/><br/>

## Active learning
See [maskAL.py](maskAL.py)
<br/> <br/>

## License
Our software was forked from detectron2 (https://github.com/facebookresearch/detectron2). As such, the software will be released under the [Apache 2.0 license](LICENSE). <br/><br/>

## Acknowledgements
Two of the software methods were inspired by the work of RovelMan (https://github.com/RovelMan/active-learning-framework). maskAL was developed by Pieter Blok (pieter.blok@wur.nl).<br/><br/>
