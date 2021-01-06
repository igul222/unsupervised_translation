import procedures.supervised

args = procedures.supervised.make_args()
args.dataset = 'mnist_usps'
args.hparam_search = True
args.unwhitened = True
procedures.supervised.main(args)