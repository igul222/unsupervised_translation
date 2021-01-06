import procedures.dann

args = procedures.dann.make_args()
args.hparam_search = True
args.prediction_method = 'top'
args.z_dim = 64
procedures.dann.main(args)