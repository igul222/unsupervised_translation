import procedures.dann

args = procedures.dann.make_args()
args.hparam_search = True
args.prediction_method = 'expectation'
args.z_dim = 16
procedures.dann.main(args)