import procedures.dann

args = procedures.dann.make_args()
args.dataset = 'mnist_usps'
args.hparam_search = True
args.prediction_method = 'expectation'
args.unwhitened = True
args.z_dim = 32
procedures.dann.main(args)