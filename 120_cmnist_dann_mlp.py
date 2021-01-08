import procedures.dann

args = procedures.dann.make_args()
args.lr_g = 1e-4
args.rep_network = 'mlp'
args.z_dim = 128
procedures.dann.main(args)