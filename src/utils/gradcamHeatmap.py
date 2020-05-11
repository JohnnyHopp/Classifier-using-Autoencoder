from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable



class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_block, target_layer):
        self.model = model
        self.target_block = target_block
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.encoder.encoder._modules.items():
            
            if int(module_pos) == self.target_block:
#            if int(module_pos) == self.target_layer:
                for sub_module_pos, sub_module in module._modules.items():
#                    print(sub_module)
                    x = sub_module(x)
                    if int(sub_module_pos) == self.target_layer:                        
                        x.register_hook(self.save_gradient)
                        conv_output = x  # Save the convolution output on that layer
                break
            x = module(x)  # Forward
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.clf(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_block, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_block, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 10)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.encoder.zero_grad()
        self.model.clf.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam

def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((32, 32), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
#        im_as_arr[channel] -= mean[channel]
#        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var
        
def get_example_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    example_list = (('./test_images/airplane1.png', 0),
                    ('./test_images/airplane5.png', 0),
                    ('./test_images/airplane6.png', 0),
                    ('./test_images/automobile1.png', 1),
                    ('./test_images/automobile2.png', 1),
                    ('./test_images/automobile3.png', 1),
                    ('./test_images/automobile5.png', 1),
                    ('./test_images/bird1.png', 2),
                    ('./test_images/bird2.png', 2),
                    ('./test_images/bird3.png', 2),
                    ('./test_images/bird6.png', 2),
                    ('./test_images/bird9.png', 2),
                    ('./test_images/cat1.png', 3),
                    ('./test_images/cat2.png', 3),
                    ('./test_images/cat3.png', 3),
                    ('./test_images/cat4.png', 3),
                    ('./test_images/cat5.png', 3),
                    ('./test_images/deer1.png', 4),
                    ('./test_images/dog1.png', 5),
                    ('./test_images/dog2.png', 5),
                    ('./test_images/dog3.png', 5),
                    ('./test_images/dog4.png', 5),
                    ('./test_images/dog5.png', 5),
                    ('./test_images/dog6.png', 5),
                    ('./test_images/dog7.png', 5),
                    ('./test_images/horse1.png', 7),                    
                    ('./test_images/ship1.png', 8),
                    ('./test_images/ship2.png', 8),
                    ('./test_images/ship3.png', 8),
                    ('./test_images/ship4.png', 8),                   
                    ('./test_images/truck1.png', 9),
                    ('./test_images/truck2.png', 9),
                    ('./test_images/truck3.png', 9)
                    )
    img_path = example_list[example_index][0]
#    target_class = example_list[example_index][1]
    target_class = None
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    # Read image
    original_image = Image.open(img_path).convert('RGB')
    # Process image
    prep_img = preprocess_image(original_image)
    # Define model
#    pretrained_model = models.alexnet(pretrained=True)
    
    return (original_image,
            prep_img,
            target_class,
            file_name_to_export)
    
