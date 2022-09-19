# Troubleshooting

Below is a list of common errors and their possible solutions:<br/><br/>

**CUDA related errors:**  <br/>

```python
d2.utils.memory Attempting to copy inputs of <function paste_masks_in_image at ...> to CPU due to CUDA OOM
```
If this error occurs during the sampling of the most uncertain images, it might indicate that your GPU has memory issues when sampling the images. A possible solution is to reduce the number of Monte-Carlo iterations (**mcd_iterations**) and/or the pool size (**pool_size**) in your configuration file (maskAL.yaml).<br/><br/>


**Error in function check_json_presence:**  <br/>

```python
active_learning.sampling.prepare_dataset - ERROR - Error in function check_json_presence: 
FileNotFoundError(2, 'No such file or directory')
```

This error can occur when the network_config or pretrained_weights is not specified correctly in the maskAL.yaml file (when auto_annotate is set to True). The Detectron2-code assumes that you are using relative file-paths instead of absolute file-paths. For example: the pretrained COCO-weights always need to be specified as COCO-InstanceSegmentation/[...].yaml and not ~/maskal/configs/COCO-InstanceSegmentation/[...].yaml.

If you are using the Windows operating system then specify the yaml file with a forward slash (COCO-InstanceSegmentation/[...].yaml), otherwise an error will be thrown. When you want to use a specific pretrained weights-file under Windows then use two backward slashes (for example: C:\\\\Users\\\\PieterBlok\\\\Documents\\\\maskal\\\\weights\\\\my_own_weights.pth).<br/><br/>
