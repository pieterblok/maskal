# Troubleshooting

Below is a list of common errors and their possible solutions:<br/><br/>

**CUDA related errors:**  <br/>

```python
d2.utils.memory Attempting to copy inputs of <function paste_masks_in_image at ...> to CPU due to CUDA OOM
```
If this error occurs during the sampling of the most uncertain images, it might indicate that your GPU has memory issues when sampling the images. A possible solution is to reduce the number of Monte-Carlo iterations (**mcd_iterations**) and/or the pool size (**pool_size**) in your configuration file (maskAL.yaml).

