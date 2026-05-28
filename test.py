from option_test import args
import torch
import utility
import dataload
import model
import loss
from tester import Tester
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def calc_params(model, res=False):
    from thop import profile
    from thop import clever_format

    x = (
        torch.randn(1,  3, 192,192).cuda(), 
        torch.randn(1,  3, 768, 768).cuda(), 
        True
    )
    
    inp = (x, ) 

    macs, params = profile(model.cuda(), inputs=inp)
    macs, params = clever_format([macs, params], "%.3f")
    print(f'Params(M): {params}, FLOPs(G): {macs}')
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params(M)": total_params / 1e6,
        "trainable_params(M)": trainable_params / 1e6
    }

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)

    if checkpoint.ok:
        loader = dataload.Data(args, 'test')
        model = model.Model(args, checkpoint)
        #calc_params(model)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Tester(args, loader, model, checkpoint)
        t.test(args.resume)

        checkpoint.done()