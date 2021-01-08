import procedures.dann

args = procedures.dann.make_args()
args.lr_g = 1e-4
args.rep_network = 'cnn'
args.steps = 5001
procedures.dann.main(args)