class Config():
    def __init__(self, 
        batch_size=32,
        cuda = False,
        gpu_id = 0,
        is_parallel = False,
        dir_data_root = "./data",
        dir_log = "./logs",
        dir_model = "/disk1/haohy/model_save",
        path_results="./results", 
        dataset_name = 'penn',
        log="log_n", 
        seq_len = 80,
        valid_len = 40,
        optim = 'Adam',
        lr = 1e-4,
        is_corpus = True,
        seed=1111,
        nhid = 600,
        levels = 4,
        emsize = 600,
        num_subblocks=2, 
        temp_attn = True,
        en_res = True,
        nheads = 1,
        dropout = 0.45,
        emb_dropout=0.25, 
        ksize=3, 
        key_size=600,
        tied=True,
        epochs=150,
        clip=0.35,
        continue_train=False,
        num_workers=1,
        permute=False,
        visual=True,
        conv=True,
        vhdropout=0):

        self.batch_size=batch_size  
        self.num_subblocks=num_subblocks 
        self.cuda=cuda  
        self.dropout=dropout 
        self.emb_dropout=emb_dropout 
        self.clip=clip 
        self.epochs=epochs 
        self.ksize=ksize 
        self.dir_data_root=dir_data_root 
        self.dir_model=dir_model 
        self.emsize=emsize 
        self.levels=levels 
        self.lr=lr 
        self.nhid=nhid 
        self.seed=seed 
        self.tied=tied 
        self.optim=optim 
        self.log=log 
        self.valid_len=valid_len 
        self.seq_len=seq_len 
        self.is_corpus=is_corpus
        self.path_results=path_results
        self.key_size = key_size
        self.gpu_id = gpu_id
        self.nheads = nheads
        self.en_res = en_res
        self.temp_attn = temp_attn
        self.is_parallel = is_parallel
        self.dir_log = dir_log
        self.dataset_name = dataset_name
        self.vhdropout = vhdropout
        self.continue_train = continue_train
        self.num_workers = num_workers
        self.permute = permute
        self.visual = visual
        self.conv = conv

config = Config()
