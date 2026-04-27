import sys, argparse, torch, numpy as np, random
p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
args, rest = p.parse_known_args()
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed); random.seed(args.seed)
sys.argv = [sys.argv[0]] + rest
exec(open("/workspace/train_distill_pf_pm.py").read())
