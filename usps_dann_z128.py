import procedures.dann

args = procedures.dann.make_args()
args.dataset = 'mnist_usps'
args.hparam_search = True
args.unwhitened = True
args.z_dim = 128
procedures.dann.main(args)