import procedures.dann

args = procedures.dann.make_args()
args.z_dim = 8
args.lambda_gp = 0.1
procedures.dann.main(args)