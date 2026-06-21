"""Matched 3-seed control: 3-seed FT-Lange ENSEMBLE (logit-mean) vs PFWL3S 3-seed vs PM,
d=7, 100K shots, p in {0.007,0.010,0.015}. Addresses the ensemble-vs-single-model fairness
gap: PFWL3S (3-seed) was previously compared only to a single Lange GNN. Here Lange is also
a 3-seed logit-mean ensemble of fine-tuned (30-epoch @ p=0.007) seeds, matched to PFWL3S."""
import sys, os, glob, json, math
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/pf/train')
sys.path.insert(0, '/workspace/GNN_decoder')
import numpy as np, torch, pymatching
from torch_geometric.nn import knn_graph
from ensemble_pf_lange import LangeWrapper, PathfinderMapper, wilson, make_circuit
from model import NeuralDecoder
device = torch.device('cuda')

class LangeLogits(LangeWrapper):
    """Same as LangeWrapper but returns per-shot LOGITS (for ensembling) and can load an FT ckpt."""
    def load_ft(self, path):
        ck = torch.load(path, weights_only=False, map_location=device)
        self.model.load_state_dict(ck['model']); self.model.eval(); return self
    def logits_batch(self, det):
        B = det.shape[0]
        lg = np.full((B,), -30.0, dtype=np.float32)         # trivial syndrome -> predict 0
        any_flip = np.sum(det, axis=1) != 0
        if not np.any(any_flip): return lg
        det_nt = det[any_flip]
        s3d = self.stim_to_3d(det_nt).astype(np.float32)
        inds = np.nonzero(s3d); defs = s3d[inds]; inds_t = np.transpose(np.array(inds))
        x_def = defs == 1; z_def = defs == 3
        nf = np.zeros((defs.shape[0], 6), dtype=np.float32)
        nf[x_def, 0] = 1; nf[x_def, 2:] = inds_t[x_def, ...]
        nf[z_def, 1] = 1; nf[z_def, 2:] = inds_t[z_def, ...]
        x = torch.tensor(nf[:, [0,1,3,4,5]]).to(device)
        batch = torch.tensor(nf[:, 2]).long().to(device); pos = x[:, 2:]
        ei = knn_graph(pos, self.m, batch=batch)
        dist = torch.sqrt(((pos[ei[0]] - pos[ei[1]])**2).sum(1, keepdim=True))
        edge_attr = 1.0 / (dist ** self.power)
        with torch.no_grad():
            out = self.model(x, ei, batch, edge_attr).cpu().numpy().reshape(-1)  # GNN_7.forward(x, edge_index, batch, edge_attr)
        lg[any_flip] = out
        return lg

PF_CKPTS = [f'/workspace/persist/checkpoints/pathfinder_wide_long_d7{s}/best_model.pt' for s in ['','_seed1','_seed2']]
def load_pf():
    ms=[]
    for p in PF_CKPTS:
        ck=torch.load(p, weights_only=False, map_location=device)
        m=NeuralDecoder(ck['config']).to(device); m.load_state_dict(ck['model_state_dict']); m.eval(); ms.append(m)
    return ms
def pf_logits(models, syn):
    with torch.no_grad():
        avg=None
        for m in models:
            lg=m(syn).cpu().numpy().reshape(-1)
            avg=lg if avg is None else avg+lg
    return avg/len(models)

d=7; NOISE=[0.007,0.010,0.015]; SEEDS=[3000,3001,3002,3003,3004]; NPS=20000
def mcnemar(b,c):
    # discordant b,c; continuity-corrected chi2(1) + survival p
    chi=(abs(b-c)-1)**2/(b+c) if (b+c)>0 else 0.0
    p=math.erfc(math.sqrt(chi/2)) if chi>0 else 1.0
    return chi,p

def main():
    pf=load_pf(); print(f"loaded {len(pf)} PFWL3S")
    # FULL-RECIPE 3-seed FT-Lange ensemble: existing seed0 (2M x30, the C2 2.739% ckpt) + 2 new seeds.
    ft_paths=[os.environ.get('FT_SEED0','/workspace/lange_finetuned_d7_p007.pt')]
    ft_paths+=[sorted(glob.glob(f'/workspace/ftcfg/seed{i}/saved_models/d_7/*model.pt'))[-1] for i in (1,2)]
    print("FT-Lange ckpts (full recipe):", [os.path.basename(p) for p in ft_paths])
    lange_ft=[LangeLogits(d,d) for _ in range(3)]
    out={'note':'FULL-RECIPE matched 3-seed: PFWL3S-3seed vs Lange-3seed-FT-ensemble (2M x30 each) vs Lange-pub vs PM, 100K shots; with McNemar paired test','rates':{}}
    for p in NOISE:
        c=make_circuit(d,p); pfm=PathfinderMapper(c)
        lw_pub=LangeLogits(d,d); lw_pub.init_from_circuit(c)
        for lw in lange_ft: lw.init_from_circuit(c)
        for lw,fp in zip(lange_ft, ft_paths): lw.load_ft(fp)
        pf_e=lpub_e=lft_e=pm_e=tot=0
        # McNemar contingency PFWL3S vs Lange-FT-ensemble: b=PF wrong&Lange right, cc=PF right&Lange wrong
        b_pfw_lr=cc_pfr_lw=both_w=0
        for ss in SEEDS:
            det,obs=c.compile_detector_sampler(seed=ss).sample(shots=NPS,separate_observables=True)
            det=det.astype(np.uint8); obs=obs.astype(np.uint8).reshape(-1)
            dem=c.detector_error_model(decompose_errors=True)
            pm_pr=pymatching.Matching.from_detector_error_model(dem).decode_batch(det).astype(np.uint8).reshape(-1)
            pf_lg=np.zeros(NPS,np.float32)
            for i in range(0,NPS,1000):
                syn=pfm.to_tensor(det[i:i+1000]).to(device); pf_lg[i:i+1000]=pf_logits(pf,syn)
            pf_pr=(pf_lg>0).astype(np.uint8)
            lpub_pr=(lw_pub.logits_batch(det)>0).astype(np.uint8)
            lft_avg=np.mean([lw.logits_batch(det) for lw in lange_ft],axis=0)
            lft_pr=(lft_avg>0).astype(np.uint8)
            pf_wrong=pf_pr!=obs; lft_wrong=lft_pr!=obs
            b_pfw_lr+=int((pf_wrong & ~lft_wrong).sum()); cc_pfr_lw+=int((~pf_wrong & lft_wrong).sum()); both_w+=int((pf_wrong & lft_wrong).sum())
            pf_e+=int(pf_wrong.sum()); lpub_e+=int((lpub_pr!=obs).sum())
            lft_e+=int(lft_wrong.sum()); pm_e+=int((pm_pr!=obs).sum()); tot+=NPS
        def ci(e): return [e/tot]+list(wilson(e,tot))
        chi,mp=mcnemar(b_pfw_lr,cc_pfr_lw)
        pf_ci=ci(pf_e); lft_ci=ci(lft_e)
        # ci() = [rate, rate, lo, hi] (wilson returns rate,lo,hi). strict win = PF_hi < Lange_lo.
        pf_lo,pf_hi=pf_ci[2],pf_ci[3]; lf_lo,lf_hi=lft_ci[2],lft_ci[3]
        marg='PFWL3S_strict' if pf_hi<lf_lo else ('Lange_strict' if lf_hi<pf_lo else 'overlap')
        r={'n':tot,'PFWL3S_3seed':ci(pf_e),'Lange_pub_1seed':ci(lpub_e),'Lange_FT_3seed':ci(lft_e),'PM':ci(pm_e),
           'mcnemar_PFWL3S_vs_LangeFT':{'b_pf_wrong_lange_right':b_pfw_lr,'c_pf_right_lange_wrong':cc_pfr_lw,'both_wrong':both_w,'chi2':chi,'p':mp},
           'PFWL3S_vs_LangeFT3seed_marginalCI':marg}
        out['rates'][f'p{p}']=r
        print(f"p={p}: PFWL3S {pf_e/tot*100:.3f}%  Lange-FT-3seed {lft_e/tot*100:.3f}%  Lange-pub {lpub_e/tot*100:.3f}%  PM {pm_e/tot*100:.3f}%  | marginalCI={r['PFWL3S_vs_LangeFT3seed_marginalCI']}  McNemar chi2={chi:.2f} p={mp:.4f} (PF-wins={cc_pfr_lw} Lange-wins={b_pfw_lr})",flush=True)
    json.dump(out,open('/workspace/lange_3seed_eval.json','w'),indent=2)
    print("saved /workspace/lange_3seed_eval.json")
if __name__=='__main__': main()
