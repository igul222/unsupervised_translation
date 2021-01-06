import procedures.dann

args = procedures.dann.make_args()
args.hparam_search = True
args.l2reg_c = 0.
args.linear_classifier = True
args.z_dim = 32
procedures.dann.main(args)