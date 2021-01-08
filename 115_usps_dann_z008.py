import procedures.dann

args = procedures.dann.make_args()
args.dataset = 'mnist_usps'
args.unwhitened = True
args.z_dim = 8
args.lambda_gp = 0.1
args.lambda_erm = 10.
args.l2reg_c = 1e-1
args.l2reg_d = 1e-3
procedures.dann.main(args)