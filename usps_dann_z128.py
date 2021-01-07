import procedures.dann

args = procedures.dann.make_args()
args.dataset = 'mnist_usps'
args.unwhitened = True
args.z_dim = 128
args.l2reg_c = 1e-3
args.l2reg_d = 1e-3
procedures.dann.main(args)