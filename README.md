# Autozoom-Attack
Codes for reproducing query-efficient black-box attacks in  “AutoZOOM: Autoencoder-based Zeroth Order Optimization Method for Attacking Black-box Neural Networks” ​​​​​​


# Software version
The program is developed under Tensorflow 1.6.0 and Tensorflow-gpu 1.6.0. Note that we use Keras embedded inside Tensorflow. It is highly recommended to use the exact same version of Tensorflow and Tensorflow-gpu


# Datasets
The dataset **mnist** and **cifar10** would be downloaded automatically the first time you use them. For the **imagenet** dataset please download the file using the following link:

[ImageNet Test images](http://www-personal.umich.edu/~timtu/Downloads/imagenet_npy/imagenet_test_data.npy)

[ImageNet Test labels](http://www-personal.umich.edu/~timtu/Downloads/imagenet_npy/imagenet_test_labels.npy)

The images and labels are stored with numpy format. Please download them and put them under AutoZOOM folder. We use the class `ImageNetDataNp` defined under file `setup_inception.py` to load these two files.

# Classifiers
We provide the classifiers for **mnist** and **cifar10** under the folder `models`.  See `setup_mnist.py` and `setup_cifar.py` for more information about the classifiers. To obtain the classifier for **imagenet** run

```
python3 setup_inception.py
```
This will download the inception_v3 model pre-trained for **imagenet**.

# Autoencoder
We provide the autoencoder for **mnist**, **cifar10** and **imagenet**.The naming for encoder and decoder are *\<dataset>_<compress_mode>_encoder.h5* and *\<dataset>_<compress_mode>_decoder.h5*. The *compress_mode* is an integer indicating the compression rate of the additive noise (attack space). When *compress_mode*=1, the width and height are respectively reduced by 1/2 of the original size. For *compress_mode*=2, its width and height are reduced to 1/4. As for **imagenet**, the settings are slightly different. Since the original image size for **imagenet** is 299x299, we first resize the image to 128x128 and then compress the image.

See `setup_codec.py` for more information about the autoencoder.


# Run attacks
Several options can be used to configure the attacks:

## Arguments

### Attack methods

We provide four kinds of attacking methods: **zoo**, **zoo_ae**, **zoo_rv** and **autozoom**.
The method **zoo** uses the method proposed in [ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks without Training Substitute Models](https://arxiv.org/abs/1708.03999). For **zoo_ae**, autoencoders are used to reduce the size of attack space. Random vectors are used in **zoo_rv** for gradient estimation. In **autozoom**, both autoencoder and random vector are used for efficient blackbox attack. The attack method can be specified using `-a` or `--attack_method`.

### Batch size
For **zoo** and **zoo_ae**, attacks are performed in batch. Use `-b` or `--batch_size` to specify the number of pixels updated within one iteration. For **zoo_rv**, **autozoom**, this option is not valid.

### Initial constant and switch iterations
The attacking procedure uses a constant to determine the preference between the likelihood of a success attack and the distortion of the image. A large constant results in fast attack (less iteration) but also indicates large distortion. This constant will be adjusted after several iterations. To specify the initial setting of the constant, use `-c` or `--init_const`. The number of iterations to update the constant can be specified using `--switch_iterations`.


### Attack space
For **zoo** and **zoo_rv**, you can specify the size on the attacking space using `--img_resize`. For **zoo_ae** and **autozoom**  the size of the attacking space is defined by the autoencoder.

### Dataset
Specify the dataset using `-d` or `--dataset`.

### Number of images to attack
Use `-n` or `--num_img` to specify the number of images to attack. Images that are correctly classified by the classifier would be randomly selected.

### Maximum number of iterations
Use `-m` or `--max_iterations` to specify the number of the maximum iteration for attacking one image. 

### Attack type
Two kinds of attack types can be specified: `targeted` and `untargeted`.

### Codec prefix
This argument should be provided when using **zoo_ae** or **autozoom**. The program will load the codec based on this argument. See the example below.

## Examples
Here we provide several examples 

1.

```
python3 main.py -a zoo -d mnist -n 100 --m 1000 \
    --batch_size 128 --switch_iterations 100 \
    --init_const 10 --img_resize 14
```

This will attack 100 images of the **mnist** dataset using the **zoo** method with batch size set to 128. The regularization constant is initialized to 10 and will be updated every 100 iterations. Finally, the attack space is reduced to 14x14 (the original size is 28x28).


2.

```
python3 main.py -a zoo_ae -d cifar10 -n 100 --m 1000 \
 --switch_iterations 1000 --init_const 10 \
 --codec_prefix codec/cifar10_2
```

This will attack 100 images of the **cifar10** dataset using the **zoo_ae** method. We specify the codec prefix as `codec/cifar10_2` so that both `codec/cifar10_2_encoder.h5` and `codec/cifar10_2_decoder.h5`
 will be loaded. 


3. 

```
python3 main.py -a autozoom -d imagenet -n 1 --img_offset 9 \
 --m 100000 --switch_iterations 1000 --init_const 10 \
 --codec_prefix codec/imagenet_3 --random_target
```

This will attack the 10-th (index starts from 0) image of the **imagenet** dataset using the **autozoom** method. We specify the codec prefix as `codec/imagenet_3` so that both `codec/imagenet_3_encoder.h5` and `codec/imagenet_3_decoder.h5` will be loaded. 