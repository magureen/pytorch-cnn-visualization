# pytorch-cnn-visualization
Easy-to-use visualization library for Grad-CAM, Guided Backpropagation, Guided Grad-CAM


# Usage
Import library and utility functions
```
from cnn_visualize import GradCam, SaliencyMap
from cnn_visualize import image_with_colormap, convert_to_grayscale, get_positive_negative_saliency
```

import library
```
from cnn_visualize import GradCam, SaliencyMap
from cnn_visualize import image_with_colormap, convert_to_grayscale, get_positive_negative_saliency
```

Prepare image
```
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
transform = Compose([
                     Resize((224,224)),
                     ToTensor(),
                     Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
                     
device = 'cpu'
img = Image.open('snake.jpg')
input_img = transform(img).to(device)
plt.imshow(img)
```
![index](https://user-images.githubusercontent.com/58849368/136645934-31925f9f-930c-4d09-b48b-08345d139850.png)

Creating Grad-CAM visualizer
```
model = torchvision.models.vgg16(pretrained=True)
vis = GradCam(model)
```

Creating heatmaps and overlaps
```
cam, c = vis(input_img)
out = image_with_colormap(img, cam)
plt.imshow(out)
```
![index](https://user-images.githubusercontent.com/58849368/136646189-667bd938-46c1-4123-a311-b9e4df5d1fe6.png)

guided back propagation
```
model = timm.create_model(model_name, pretrained=True).to(device).eval()
vis = SaliencyMap(model, guided=True)
gbp = convert_to_grayscale(grad)
plt.imshow(gbp)
```
![index](https://user-images.githubusercontent.com/58849368/136646197-054e0127-72ea-45de-ad97-c7c6500e0c37.png)

positive saliency
```
pos, neg = get_positive_negative_saliency(grad)
plt.imshow(pos)
```
![index](https://user-images.githubusercontent.com/58849368/136646235-d7312671-e309-4b46-bdb0-0faa53bb9ce5.png)

negative saliency
```
plt.imshow(neg)
```
![index](https://user-images.githubusercontent.com/58849368/136646257-61c4bd3a-d8e3-4861-b7d0-69bc0608fe80.png)

guided Grad-CAM
```
plt.imshow(out * np.expand_dims(gbp, axis=2))
```
![index](https://user-images.githubusercontent.com/58849368/136646271-bb0049c1-3279-4504-9741-8c62f20e068e.png)


# References
https://github.com/utkuozbulak/pytorch-cnn-visualizations
https://github.com/FrancescoSaverioZuppichini/A-journey-into-Convolutional-Neural-Network-visualization-
