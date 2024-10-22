from model.model_proposed import MalConv_Proposed
from model.model_malconv import MalConv
from malconv2.MalConvGCT_nocat import MalConvGCT as malconv2


def get_malconv_variant(args):
    if args.variant == 'proposed':
        malconv = MalConv_Proposed(args) 
        # Freeze layers for proposed approach
        for layer_idx, layer in enumerate(malconv.parameters()):
            if layer_idx <= 4:
                layer.requires_grad = False
    elif args.variant == 'malconv2':
        # Freeze layers using MalConvGCT_nocat.py
        malconv = malconv2(out_size=args.num_classes, chunk_size=args.fp_slice_size, channels=args.num_filters, window_size=args.window_size, stride=args.stride)#.to(args.device)
    elif args.variant == 'malconv':
        malconv = MalConv(args) 
        for layer_idx, layer in enumerate(malconv.parameters()):
           if layer_idx <= 1:
               layer.requires_grad = True # False
    malconv = malconv.to(args.device)
    return malconv
    
