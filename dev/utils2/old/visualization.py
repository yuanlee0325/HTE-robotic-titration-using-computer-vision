import matplotlib.pyplot as plt
from torchvision.utils import make_grid

## visualization for LinearGAN
def show_tensor_images(image_tensor, num_images=25, size=(3, 256, 256)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    #image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.view(-1,*size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    #plt.show()
    plt.xticks([])
    plt.yticks([])