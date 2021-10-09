# pytorch-cnn-visualization
Easy-to-use visualization library for Grad-CAM, Guided Backpropagation, Guided Grad-CAM


# Usage
download
```
wget https://raw.githubusercontent.com/magureen/pytorch-cnn-visualization/main/cnn_visualize.py
```

import library and utility functions
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
model = torchvision.models.vgg16(pretrained=True)
vis = SaliencyMap(model, guided=True)
grad, c = vis(input_img)
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
plt.imshow(cam * gbp)
```
![index](https://user-images.githubusercontent.com/58849368/136659589-64fa424a-a185-460f-bfc6-fbbf7c404f2c.png)


# Use other models and specify classes
```
img = Image.open('dog_and_cat.jpg').resize((224,224))
input_img = transform(img).to(device)
plt.imshow(img)
```
![index](https://user-images.githubusercontent.com/58849368/136659608-096d7d3a-0cb8-44ac-b99f-306b127bc561.png)

Use timm model
```
model_name = 'efficientnet_b3'
model = timm.create_model(model_name, pretrained=True).to(device).eval()
vis_cam = GradCam(model)
model = timm.create_model(model_name, pretrained=True).to(device).eval()
vis_gbp = SaliencyMap(model, guided=True)
```

Class specification
```
target_class = 260 # dog
cam, c = vis_cam(input_img, target_class=target_class)
grad, c = vis_gbp(input_img, target_class=target_class)

out = image_with_colormap(img, cam)
plt.imshow(out)
plt.show()

gbp = convert_to_grayscale(grad)
plt.imshow(cam*gbp)
plt.show()
```
![index](https://user-images.githubusercontent.com/58849368/136659838-b70ffc47-c0ac-41bd-aa69-751ef199d48a.png) ![index](https://user-images.githubusercontent.com/58849368/136659846-519cacaf-42d4-4dbf-b58d-51c40775ad04.png)

```
target_class = 281 # cat
cam, c = vis_cam(input_img, target_class=target_class)
grad, c = vis_gbp(input_img, target_class=target_class)

out = image_with_colormap(img, cam)
plt.imshow(out)
plt.show()

gbp = convert_to_grayscale(grad)
plt.imshow(cam*gbp)
plt.show()
```
![index](https://user-images.githubusercontent.com/58849368/136659945-a553870e-e74d-412c-971b-9d8cc9f19d94.png) ![index](https://user-images.githubusercontent.com/58849368/136659966-36297a0f-6703-4ed0-a957-8073ced4e087.png)

# References
- https://github.com/utkuozbulak/pytorch-cnn-visualizations
- https://github.com/FrancescoSaverioZuppichini/A-journey-into-Convolutional-Neural-Network-visualization-
