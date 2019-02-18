import time
from options.train_options import TrainOptions
from options.base_options import BaseOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    # change default configuration
    opt.dataroot = '/home/Imatge/hd/datasets/insta10YearsChallenge/splits_dlib_aligned' # path to images (should have subfolders trainA, trainB, valA, valB, etc)
    opt.name = 'rejuvenating_dlib_algined_21k_pix2pix_lrfixed25then100' # name of the experiment. It decides where to store samples and models
    opt.model = 'pix2pix'
    opt.direction = 'BtoA'
    opt.gpu_ids = [3] # gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU
    # opt.ngf = 64  # default=64, # of gen filters in the last conv layer
    # opt.ndf = 64  # default=64, # of discrim filters in the first conv layer
    # opt.netD = 'basic' #  default='basic', 70x70 PatchGAN
    opt.display_id = 5

    opt.netG = 'unet_256' #  default='resnet_9blocks' specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
    # default netG for pix2pix is unet_265!

    # opt.n_layers_D = '3' #  default=3  only used if netD==n_layers'
    # opt.norm = 'instance' # default='instance' instance normalization or batch normalization [instance | batch | none]
    # default norm for pix2pix is batch norm!

    # opt.direction = 'BtoA'  # default='BtoA'
    # opt.batch_size = 1  # default=1
    # opt.preprocess = 'resize_and_crop', # default = 'resize_and_crop'
    # opt.no_flip  # if specified, do not flip the images for data augmentation

    # change default learning rate
    opt.niter = 25 # # of iter (epochs) at starting learning rate
    opt.niter_decay = 100  # of iter (epochs) to linearly decay learning rate to zero')
    opt.beta1 = 0.5  # momentum term of adam
    opt.lr = 0.0002  #  initial learning rate for adam')
    opt.lr_policy = 'linear'  # learning rate policy. [linear | step | plateau | cosine]
    opt.lr_decay_iters = 50  # multiply by a gamma every lr_decay_iters iterations (epochs)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
