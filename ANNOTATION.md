Because active learning involves an iterative annotation of the most-informative images, we assume that the training, validation and test images have their corresponding json/xml annotations from the either the **LabelMe** software, **V7-Darwin** software or **CVAT** software (when using CVAT, export the annotations to **LabelMe 3.0** format). Our default annotation procedure is based on the **LabelMe** software. Please find the installation instructions of labelme on: https://github.com/wkentaro/labelme.
<br/>

**It is not required to annotate all images yet, because the active learning algorithm will select the most-informative images for you to annotate.** <br/>

*Annotation procedure (LabelMe):*
1. Annotate each individual object by either a polygon, a circle or a rectangle. For our use-case, the polygon was the default shape.
2. Assign the correct class-name to the object. 
3. When an object consists of two or more separated parts: draw each separated part by an individual polygon and link the polygons by the same group id.
<br/> <br/> ![LabelMe annotation](./demo/labelme_annotation.png?raw=true)
<br/> *The broccoli head of the class "cateye" is occluded by a leaf, causing two separated instances of the broccoli head. Annotate the individual instances by a separate polygon and link them with an unique group_id (in this example 1). Suppose there is another occluded broccoli head in the image: then use another group_id (for example 2).*
