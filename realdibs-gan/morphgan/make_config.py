
import json


class Config:
    pass

    def import_jsonfile(self,f):

        # load file and parse
        with open(f) as f:
            json_dict = json.load(f)

        for key, value in json_dict.items():
            setattr(self, key, value)

    def export_jsonfile(self,f):

        # convert to json-able
        dt = {}
        dt.update(vars(self))

        # save to stream
        with open(f, "w") as file:
            json.dump(dt, file, indent=2, sort_keys=True)


if __name__ == "__main__":
    """ 
    Create config file.
    Change parameters for the various experiments
    """

    config = Config()
    base_path = 'configs/'

    #############################

    # Output Flags
    config.out_dir = "/mnt/soarin/results/"
    config.task = 'gan_models'

    for m in ['pix2pix', 'cyclegan']:
        config.method = m

        # Dataset Flags

        if 0:
            config.dataset_name = "sullens_morphed"
            config.height = 3000
            config.width = 4000
            config.dataset_version = '0.2.0'

        if 0:
            config.dataset_name = "trinity_crop_morphed"
            config.height = 3000
            config.width = 4000
            config.dataset_version = '0.2.0'

        if 0:
            config.dataset_name = "sullens"
            config.height = 3000
            config.width = 4000
            config.dataset_version = '0.3.0'

        if 0:
            config.dataset_name = 'ruemonge'
            config.height = 1024
            config.width = 800
            config.dataset_version = '0.3.0'

        if 0:
            config.dataset_name = 'trinity'
            config.height = 3000
            config.width = 4000

        if 0:
            config.dataset_name = 'merton3'
            config.height = 1024
            config.width = 768

        if 1:
            config.dataset_name = "sullens_morphed_animation"
            config.height = 3000
            config.width = 4000
            config.dataset_version = '0.2.0'

        config.data_dir = '/mnt/soarin/Datasets/'
        config.buffer_size = 3
        config.prefetch_size = 270

        # Training Flags
        config.gpu = "1"
        config.epochs = 10000
        config.patch_size = 768
        config.batch_size = 3
        config.checkpoint_freq = 1000
        config.lr = 2e-4
        config.initial_epoch = 5000

        # Testing Flags
        config.test_one = False
        config.test_freq = 200

        # Model Specific Flags
        config.network_downscale = 256
        config.LAMBDA = 100

        # Artgan specific flags
        config.discr_success = 0.8
        config.ngf = 32
        config.ndf = 64
        config.dlw = 1.
        config.tlw = 100.
        config.flw = 100.

        # Inference Flags
        config.ii_dir = ''
        config.save_dir = ''
        config.ckpt_path = '/mnt/soarin/results/gan_models/sullens_morphed/3_pix2pix_morphgan/' \
                           '20200225-185012/checkpoints/4999/'

        f = base_path + config.method + '_' + config.dataset_name + '_epochs_' + str(config.epochs) + '.json'
        if config.initial_epoch != 0:
            f = base_path + config.method + '_' + config.dataset_name + '_epochs_' + str(config.epochs) + \
                '_started_at_' + str(config.initial_epoch) + '.json'

        if config.ckpt_path != '':
            f = base_path + config.method + '_' + config.dataset_name + '_test_at_epoch_' + str(config.initial_epoch)\
                + '.json'

        # comment out if you want to create a specific config name
        # f = base_path + method + '_' + dataset_name + '.json'
        # f = base_path + 'config_to_gen_dataset.json'

        config.export_jsonfile(f)
        print('wrote', f)
