from argparse import ArgumentParser

from model.trainer import CL_Trainer

def pasre_arg():
    
    cfg = ArgumentParser(description = "CHU CL Framework",
                            allow_abbrev = False)
    
    # CL Experiments Settings
    cfg.add_argument('--dataset', type = str, default = 'mnist',
                     help = 'experiment dataset', choices = ['mnist', 'cifar10', 'cifar100'])
    cfg.add_argument('--image_shape', type = set, default = (32,32),
                     help = 'image_shpae of dataset')
    cfg.add_argument('--cl_type', type = str, default = 'cil',
                     help = 'CL exepriment type', choices = ['cil', 'dil'])
    cfg.add_argument('--nclasses', type = int, default = 10,
                     help = 'nclasses of dataset')
    cfg.add_argument('--num_increments', type = int, default = 5,
                     help = 'task num of CL')
    cfg.add_argument('--device', type = str, default='cuda',
                     help = 'deep learing devices', choices = ['cpu', 'cuda'])
    cfg.add_argument('--epoch', type = int, default = 10,
                     help = 'epochs per task')
    cfg.add_argument('--batch_size', type = int, default = 64,
                     help = 'batch size for current data stream, when incremental learning is adoptted, \
                         total batch size is batch_size + buffer_extraction_size in ER series')
    cfg.add_argument('--buffer_memory_size', type = int, default = 500,
                     help = 'buffer size')
    
    cfg.add_argument('--model', type = str, default = 'er',
                     help = 'cl method for continual learning(not Exemplar Storage & Extraction Method)', choices = ['er', 'er_ace', 'der', 'der++'])
    cfg.add_argument('--buffer_extraction', type = str, default = 'random',
                     help = 'buffer extraction strategy for continual learning', choices = ['random', 'mir'])
    cfg.add_argument('--buffer_extraction_size', type = int, default = 64,
                     help = 'buffer extraction size for buffer batch')
    cfg.add_argument('--buffer_storage', type = str, default = 'random',
                     help = 'buffer storage strategy for continual learning', choices = ['random', 'gss'])
    cfg.add_argument('--buffer_storage_size', type = int, default = 64,
                     help = 'buffer storage size for buffer update')
    
    cfg.add_argument('--optim', type = str, default = 'sgd',
                     help = 'optimizer for learning', choices = ['sgd'])
    cfg.add_argument('--lr', type = float, default = 1e-3,
                     help = 'learning rate for optimizer')
    cfg.add_argument('--momentum', type = float, default = 9e-1,
                     help = 'mentum for optimizer')
    cfg.add_argument('--alpha', type = float, default = 1e-1,
                     help = 'hyper parameter for knowledge distillation in DER, DER++')
    cfg.add_argument('--beta', type = float, default = 5e-1,
                    help = 'hyper parameter for knowledge distillation in DER++')
    
    # check CL tasks
    temp = cfg.parse_args()
    if temp.nclasses % temp.num_increments != 0:
        raise "cant divide into CL tasks!"
    
    return cfg.parse_args()


if __name__ == '__main__':
    
    cfg = pasre_arg()
    cl_trainer = CL_Trainer(cfg)
    
    cl_trainer.begin_continual_learning()