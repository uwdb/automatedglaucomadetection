import os


class ukbb_gl_constants:
    def __init__(self, modelnm, fold=None,transfer=False, aux_chkpoint='',
                tr=None, slice=None):
        self.model_name = modelnm
        self.base_dir = ''

        self.tr = tr
        # location of model related output files
        self.model_dir = self.base_dir
        self.save_path = self.base_dir + self.model_name + '/'
        self.data_dir = self.base_dir

        # make dirs
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.te_gl_nm = 'test_gl_nm.csv'
        self.tr_gl_nm = 'train_gl_nm.csv'
        self.val_gl_nm = 'val_gl_nm.csv'

        # root dir for data
        self.root_dir = ''

        # checkpoint+path
        self.checkpoint_path = self.save_path + 'chkpoint/cp.ckpt'
        self.weights_path = self.save_path + 'weights/final.ckpt'
        self.pb_path = self.save_path + 'pb/'
        self.num_steps = 0
        self.val_steps = 0
        self.logs = self.save_path + self.model_name + '_logs'
        self.epochs = 200
        self.batch_size = 175
        self.eval_batch_size = 80
        self.results_fn = self.save_path + self.model_name + '_results.txt'
        self.results_csv = self.save_path + self.model_name + "_output.csv"


        if not os.path.exists(self.save_path + 'chkpoint/'):
            os.mkdir(self.save_path + 'chkpoint/')
        if not os.path.exists(self.save_path + 'weights/'):
            os.mkdir(self.save_path + 'weights/')
        if not os.path.exists(self.save_path + 'pb/'):
            os.mkdir(self.save_path + 'pb/')
