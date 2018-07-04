## main.py -- sample code to test attack procedure
##
## Copyright (C) 2018, IBM Corp
##                     Chun-Chen Tu <timtu@umich.edu>
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import os
import sys
import random
import time
import copy
import numpy as np
import tensorflow as tf
import scipy.misc
from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel, ImageNetDataNP

from setup_codec import CODEC

import Utils as util
from blackbox_attack import ZOO, ZOO_AE, ZOO_RV, AutoZOOM
import argparse

def main(args):
    with tf.Session() as sess:
        print("Loading data and classification model: {}".format(args["dataset"]))
        if args['dataset'] == "mnist":
            data, model =  MNIST(), MNISTModel("models/mnist", sess, use_softmax=True)
        elif args['dataset'] == "cifar10":
            data, model = CIFAR(), CIFARModel("models/cifar", sess, use_softmax=True)
        elif args['dataset'] == "imagenet":
            # data, model = ImageNet(data_path=args["imagenet_dir"], targetFile=args["attack_single_img"]), InceptionModel(sess, use_softmax=True)
            data, model = ImageNetDataNP(), InceptionModel(sess, use_softmax=True)
        elif args['dataset'] == "imagenet_np":
            data, model = ImageNetDataNP(), InceptionModel(sess, use_softmax=True)


        if len(data.test_labels) < args["num_img"]:
            raise Exception("No enough data, only have {} but need {}".format(len(data.test_labels), args["num_img"]))


        if args["attack_single_img"]:
            # manually setup attack set
            # attacking only one image with random attack]
            orig_img = data.test_data
            orig_labels = data.test_labels
            orig_img_id = np.array([1])

            if args["attack_type"] == "targeted":
                target_labels = [np.eye(model.num_labels)[args["single_img_target_label"]]]
            else:
                target_labels = orig_labels
        else:
            # generate attack set
            if args["dataset"] == "imagenet" or args["dataset"] == "imagenet_np":
                shift_index = True
            else:
                shift_index = False

        if args["random_target"] and (args["dataset"] == "imagenet" or args["dataset"] == "imagenet_np"):
            # find all possible class
            all_class = np.unique(np.argmax(data.test_labels, 1))
            all_orig_img, all_target_labels, all_orig_labels, all_orig_img_id = util.generate_attack_data_set(data, args["num_img"], args["img_offset"], model, attack_type=args["attack_type"], random_target_class=all_class, shift_index=shift_index)
        elif args["random_target"]:
            # random target on all possible classes
            class_num = data.test_labels.shape[1]
            all_orig_img, all_target_labels, all_orig_labels, all_orig_img_id = util.generate_attack_data_set(data, args["num_img"], args["img_offset"], model, attack_type=args["attack_type"], random_target_class=list(range(class_num)), shift_index=shift_index)
        else:
            all_orig_img, all_target_labels, all_orig_labels, all_orig_img_id = util.generate_attack_data_set(data, args["num_img"], args["img_offset"], model, attack_type=args["attack_type"], shift_index=shift_index)

                # check attack data
        # for i in range(len(orig_img_id)):
        #     tar_lab = np.argmax(target_labels[i])
        #     orig_lab = np.argmax(orig_labels[i])
        #     print("{}:, target label:{}, orig_label:{}, orig_img_id:{}".format(i, tar_lab, orig_lab, orig_img_id[i]))



        # attack related settings
        if args["attack_method"] == "zoo" or args["attack_method"] == "zoo_rv":
            if args["img_resize"] is None:
                args["img_resize"] = model.image_size
                print("Argument img_resize is not set and not using autoencoder, set to image original size:{}".format(args["img_resize"]))


        if args["attack_method"] == "zoo" or args["attack_method"] == "zoo_ae":
            if args["batch_size"] is None:
                args["batch_size"] = 128
                print("Using zoo or zoo_ae attack, and batch_size is not set.\nSet batch_size to {}.".format(args["batch_size"]))
            
        else:
            if args["batch_size"] is not None:
                print("Argument batch_size is not used")
                args["batch_size"] = 1 # force to be 1

        if args["attack_method"] == "zoo_ae" or args["attack_method"] == "autozoom":
            #_, decoder = util.load_codec(args["codec_prefix"])
            if args["dataset"] == "mnist" or args["dataset"] == "cifar10":
                codec = CODEC(model.image_size, model.num_channels, args["compress_mode"], use_tanh=False)
            else:
                codec = CODEC(128, model.num_channels, args["compress_mode"])
            print(args["codec_prefix"])
            codec.load_codec(args["codec_prefix"])
            decoder = codec.decoder
            print(decoder.input_shape)
            args["img_resize"] = decoder.input_shape[1]
            print("Using autoencoder, set the attack image size to:{}".format(args["img_resize"]))

        # setup attack
        if args["attack_method"] == "zoo":
            blackbox_attack = ZOO(sess, model, args)
        elif args["attack_method"] == "zoo_ae":
            blackbox_attack = ZOO_AE(sess, model, args, decoder)
        elif args["attack_method"] == "zoo_rv":
            blackbox_attack = ZOO_RV(sess, model, args)
        elif args["attack_method"] == "autozoom":
            blackbox_attack = AutoZOOM(sess, model, args, decoder, codec)


        save_prefix = os.path.join(args["save_path"], args["dataset"], args["attack_method"], args["attack_type"])

        os.system("mkdir -p {}".format(save_prefix))

        total_success = 0
        l2_total = 0

        
        for i in range(all_orig_img_id.size):
            orig_img = all_orig_img[i:i+1]
            target = all_target_labels[i:i+1]
            label = all_orig_labels[i:i+1]

            target_class = np.argmax(target)
            true_class = np.argmax(label)
            test_index = all_orig_img_id[i]

            # print information
            print("[Info][Start]{}: test_index:{}, true label:{}, target label:{}".format(i, test_index, true_class, target_class))
            if args["attack_method"] == "zoo_ae" or args["attack_method"] == "autozoom":
                #print ae info
                if args["dataset"] == "mnist" or args["dataset"] == "cifar10":
                    temp_img = all_orig_img[i:i+1]
                else:
                    temp_img = all_orig_img[i]
                    temp_img = (temp_img+0.5)*255
                    temp_img = scipy.misc.imresize(temp_img, (128,128))
                    temp_img = temp_img/255 - 0.5
                    temp_img = np.expand_dims(temp_img, axis=0)
                encode_img = codec.encoder.predict(temp_img)
                decode_img = codec.decoder.predict(encode_img)
                diff_img = (decode_img - temp_img)
                diff_mse = np.mean(diff_img.reshape(-1)**2)
                print("[Info][AE] MSE:{:.4f}".format(diff_mse))

            timestart = time.time()
            adv_img = blackbox_attack.attack(orig_img, target)
            timeend = time.time()

            if len(adv_img.shape) == 3:
                adv_img = np.expand_dims(adv_img, axis=0)

            l2_dist = np.sum((adv_img-orig_img)**2)**.5
            adv_class = np.argmax(model.model.predict(adv_img))

            success = False
            if args["attack_type"] == "targeted":
                if adv_class == target_class:
                    success = True
            else:
                if adv_class != true_class:
                    success = True

            if success:
                total_success += 1
                l2_total += l2_dist

            print("[Info][End]{}: test_index:{}, true label:{}, adv label:{}, success:{}, distortion:{:.5f}, success_rate:{:.4f}, l2_avg:{:.4f}".format(i, test_index, true_class, adv_class, success, l2_dist, total_success/(i+1), 0 if total_success == 0 else l2_total / total_success))

            # save images
            suffix = "id{}_testIndex{}_true{}_adv{}".format(i, test_index, true_class, adv_class)
            # original image
            save_name = os.path.join(save_prefix, "Orig_{}.png".format(suffix))
            util.save_img(orig_img, save_name)
            save_name = os.path.join(save_prefix, "Orig_{}.npy".format(suffix))
            np.save(save_name, orig_img)

            # adv image
            save_name = os.path.join(save_prefix, "Adv_{}.png".format(suffix))
            util.save_img(adv_img, save_name)
            save_name = os.path.join(save_prefix, "Adv_{}.npy".format(suffix))
            np.save(save_name, adv_img)

            # diff image
            save_name = os.path.join(save_prefix, "Diff_{}.png".format(suffix))
            util.save_img((adv_img - orig_img)/2, save_name)
            save_name = os.path.join(save_prefix, "Diff_{}.npy".format(suffix))
            np.save(save_name, adv_img - orig_img)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--attack_method", default="autozoom", choices=["zoo", "zoo2", "zoo_ae", "zoo_rv", "autozoom"], help="the attack method")
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="the batch size for zoo, zoo_ae attack")
    parser.add_argument("-c", "--init_const", type=float, default=1, help="the initial setting of the constant lambda")
    parser.add_argument("-d", "--dataset", default="mnist", choices=["mnist", "cifar10", "imagenet", "imagenet_np"])
    parser.add_argument("-n", "--num_img", type=int, default=100, help = "number of test images to attack")
    parser.add_argument("-m", "--max_iterations", type=int, default=0, help = "set 0 to use default value")
    parser.add_argument("-p", "--print_every", type=int, default=100, help="print information every PRINT_EVERY iterations")
    parser.add_argument("-s", "--save_path", default=None, help="the path to save the results")
    parser.add_argument("--attack_type", default="targeted", choices=["targeted", "untargeted"], help="the type of attack")
    parser.add_argument("--confidence", default=0, type=float, help="the attack confidence")
    parser.add_argument("--codec_prefix", default=None, help="the coedec prefix, load the default codec is not set")
    parser.add_argument("--random_target", action="store_true", help="if set, choose random target, otherwise attack every possible target class, only works when ATTACK_TYPE=targeted")
    parser.add_argument("--num_rand_vec", type=int, default=1, help="the number of random vector for post success iteration")
    parser.add_argument("--seed", type=int, default=9487, help="random seed")
    parser.add_argument("--img_offset", type=int, default=0, help="the offset of the image index when getting attack data")
    parser.add_argument("--img_resize", default=None, type=int, help = "this option only works for ATTACK METHOD zoo and zoo_rv")
    parser.add_argument("--switch_iterations", type=int, default=1000, help="the iteration number for dynamic switching")
    parser.add_argument("--imagenet_dir", default=None, help="the path for imagenet images")
    parser.add_argument("--attack_single_img", default=None, help="attack a specific image, only works when DATASET is imagenet")
    parser.add_argument("--single_img_target_label", type=int, default=None, help="the target label for the single image attack")
    parser.add_argument("--compress_mode", type=int, default=None, help="specify the compress mode if autoencoder is used")

    args = vars(parser.parse_args())

    # settings based on dataset and attack method

    # mnist
    if args["dataset"] == "mnist":
        if args["max_iterations"] == 0:
            args["max_iterations"] = 1000

        args["use_tanh"] = False
        if args["codec_prefix"] is None:
            args["codec_prefix"] = "codec/mnist"

        args["lr"] = 1e-2

        args["compress_mode"] = 1

    # cifar10
    if args["dataset"] == "cifar10":
        if args["max_iterations"] == 0:
            args["max_iterations"] = 1000

        args["use_tanh"] = True
        if args["codec_prefix"] is None:
            args["codec_prefix"] = "codec/cifar10"
        args["lr"] = 1e-2
        if args["compress_mode"] is None:
            args["compress_mode"] = 2

    # imagenet
    if args["dataset"] == "imagenet" or args["dataset"] == "imagenet_np":
        if args["max_iterations"] == 0:
            if args["attack_method"] == "zoo_rv" or args["attack_method"] == "autozoom":
                args["max_iterations"] = 100000
            else:
                 args["max_iterations"] = 20000

            if not args["random_target"] and args["attack_type"] == "targeted":
                print("WARNING: You are trying to attack imagenet data with all (1000) labels.")

        if args["attack_single_img"] is not None:
            print("Imagenet targeting on one file:{}".format(args["attack_single_img"]))
            # force test image num to be 1
            args["num_img"] = 1
            if args["attack_type"] == "targeted":
                if args["single_img_target_label"] is None:
                    print("Attack target is not set.")
                    args["single_img_target_label"] = np.random.choice(range(1, 1001), 1)
                    print("Randomly choose target label:{}".format(args["single_img_target_label"]))
                else:
                    print("Targeting label:{}".format(args["single_img_target_label"]))
        # else:
        #     if args["imagenet_dir"] is None:
        #         raise Exception("Selecting imagenet as dataset but the path to the imagenet images are not set.")

        args["use_tanh"] = True
        args["lr"] = 2e-3
        
        if args["codec_prefix"] is None:
            args["codec_prefix"] = "codec/imagenet"

        args["compress_mode"] = 2

    if args["img_resize"] is not None:
        if args["attack_method"] == "zoo_ae" or args["attack_method"] == "autozoom":
            print("Attack method {} cannot use option img_resize, arugment ignored".format(args["attack_method"]))

    if args["save_path"] is None:
        # use dataset and attack method for the saving path
        args["save_path"] = "Results"



    # setup random seed
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    tf.set_random_seed(args["seed"])
    print(args)
    main(args)






