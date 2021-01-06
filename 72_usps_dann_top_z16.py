import procedures.dann

args = procedures.dann.make_args()
args.dataset = 'mnist_usps'
args.hparam_search = True
args.prediction_method = 'top'
args.unwhitened = True
args.z_dim = 16
procedures.dann.main(args)