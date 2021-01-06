import procedures.dann

args = procedures.dann.make_args()
args.hparam_search = True
args.z_dim = 128
procedures.dann.main(args)