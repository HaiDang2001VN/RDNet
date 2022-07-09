import numpy as np
import torch

class ToMask(object):
    """
        Convert a 3 channel RGB image into a 1 channel segmentation mask
        For NYUv2 dataset, the labels are presented in correct format
    """
    def __init__(self, palette_dictionnary):
        self.nb_classes = len(palette_dictionnary)
        
        # sort the dictionary of the classes by the sum of rgb value -> to have always background = 0
        # self.converted_dictionnary = {i: v for i, (k, v) in enumerate(sorted(palette_dictionnary.items(), key=lambda item: sum(item[1])))}
        
        # Use mapTolds of NYUv2 instead
        self.palette_dictionnary = palette_dictionnary

    def __call__(self, pil_image):
        # avoid taking the alpha channel
        image_array = np.array(pil_image)[:, :, :3]
        # get only one channel for the output
        output_shape = image_array.shape
        output_shape[-1] = 1
        output_array = np.zeros(output_shape, dtype="int")

        for label in self.palette_dictionnary.keys():
            rgb_color = self.palette_dictionnary[label]['color']
            mask = (image_array == rgb_color)
            output_array[mask[:, :, 0]] = int(label)

        output_array = torch.from_numpy(output_array).unsqueeze(0).long()
        return output_array
