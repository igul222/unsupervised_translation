import procedures.dann

args = procedures.dann.make_args()
args.dataset = 'mnist_usps'
args.lr_g = 1e-4
args.rep_network = 'cnn'
args.steps = 20001
args.l2reg_c = 1e-2
procedures.dann.main(args)