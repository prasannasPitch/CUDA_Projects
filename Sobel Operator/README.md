### Sobel Operator

<p align="justify">
A very common operator for edge detection is a Sobel Operator, which is an approximation to a derivative of an image. It is separate in the y and x directions. If we look at the x-direction, the gradient of an image in the x-direction is equal to this operator here. We use a kernel 3 by 3 matrix, one for each x and y direction. The gradient for x-direction has minus numbers on the left hand side and positive numbers on the right hand side and we are preserving a little bit of the center pixels. Similarly, the gradient for y-direction has minus numbers on the bottom and positive numbers on top and here we are preserving a little bit on the middle row pixels.
</p>


![parking](https://user-images.githubusercontent.com/37708330/73333992-b8f1a700-426b-11ea-9980-92ca28819e75.jpg)


![sampleSobel2](https://user-images.githubusercontent.com/37708330/73333997-ba22d400-426b-11ea-98db-988f697e06f7.png)
