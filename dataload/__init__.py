from importlib import import_module
from dataloader import MSDataLoader

class Data:
    def __init__(self, args, mode):
        if mode=='train':
            self.loader_train = None
            if not args.test_only:
                module_train = import_module('dataload.' + args.data_train.lower())     ## load the right dataset loader module
                trainset = getattr(module_train, args.data_train)(args)             ## load the dataset, args.data_train is the  dataset name
                self.loader_train = MSDataLoader(
                    args,
                    trainset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    pin_memory=False
                )
        if mode=='test':
            if args.data_test in ['Set5', 'Set14', 'BSDS100', 'manga109', 'Urban100', 'RealSRSet', 'FreeData']:
                module_test = import_module('dataload.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, name=args.data_test,train=False)
            else:
                module_test = import_module('dataload.' + args.data_test.lower())
                testset = getattr(module_test, args.data_test)(args, train=False)

            self.loader_test = MSDataLoader(
                args,
                testset,
                batch_size=1,
                shuffle=False,
                pin_memory=False
            )

