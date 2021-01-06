import procedures.dann

args = procedures.dann.make_args()
args.hparam_search = True
args.prediction_method = 'worstcase'
args.z_dim = 16
procedures.dann.main(args)