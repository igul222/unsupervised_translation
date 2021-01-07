import procedures.dann

args = procedures.dann.make_args()
args.dataset = 'mnist_usps'
args.unwhitened = True
args.z_dim = 32
args.lambda_erm = 10.
args.l2reg_c = 1e-1
args.l2reg_d = 1e-4
procedures.dann.main(args)