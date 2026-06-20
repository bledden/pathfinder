"""Matched 3-seed control: 3-seed FT-Lange ENSEMBLE (logit-mean) vs PFWL3S 3-seed vs PM,
d=7, 100K shots, p in {0.007,0.010,0.015}. Addresses the ensemble-vs-single-model fairness
gap: PFWL3S (3-seed) was previously compared only to a single Lange GNN. Here Lange is also
a 3-seed logit-mean ensemble of fine-tuned (30-epoch @ p=0.007) seeds, matched to PFWL3S."""
import sys, os, glob, json
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
def main():
    pf=load_pf(); print(f"loaded {len(pf)} PFWL3S")
    ft_paths=[sorted(glob.glob(f'/workspace/ftcfg/seed{i}/saved_models/d_7/*model.pt'))[-1] for i in range(3)]
    print("FT-Lange ckpts:", [os.path.basename(p) for p in ft_paths])
    lange_ft=[LangeLogits(d,d) for _ in range(3)]
    out={'note':'matched 3-seed: PFWL3S-3seed vs Lange-3seed-FT-ensemble vs Lange-pub vs PM, 100K shots','rates':{}}
    for p in NOISE:
        c=make_circuit(d,p); pfm=PathfinderMapper(c)
        lw_pub=LangeLogits(d,d); lw_pub.init_from_circuit(c)
        for lw in lange_ft: lw.init_from_circuit(c)
        for lw,fp in zip(lange_ft, ft_paths): lw.load_ft(fp)
        pf_e=lpub_e=lft_e=pm_e=tot=0
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
            pf_e+=int((pf_pr!=obs).sum()); lpub_e+=int((lpub_pr!=obs).sum())
            lft_e+=int((lft_pr!=obs).sum()); pm_e+=int((pm_pr!=obs).sum()); tot+=NPS
        def ci(e): return [e/tot]+list(wilson(e,tot))
        r={'n':tot,'PFWL3S_3seed':ci(pf_e),'Lange_pub_1seed':ci(lpub_e),'Lange_FT_3seed':ci(lft_e),'PM':ci(pm_e)}
        pf_ci=ci(pf_e); lft_ci=ci(lft_e)
        r['PFWL3S_vs_LangeFT3seed']=('PFWL3S_strict' if pf_ci[2]<lft_ci[1] else ('Lange_strict' if lft_ci[2]<pf_ci[1] else 'overlap'))
        out['rates'][f'p{p}']=r
        print(f"p={p}: PFWL3S {pf_e/tot*100:.3f}%  Lange-FT-3seed {lft_e/tot*100:.3f}%  Lange-pub {lpub_e/tot*100:.3f}%  PM {pm_e/tot*100:.3f}%  -> PFWL3S_vs_LangeFT3seed={r['PFWL3S_vs_LangeFT3seed']}",flush=True)
    json.dump(out,open('/workspace/lange_3seed_eval.json','w'),indent=2)
    print("saved /workspace/lange_3seed_eval.json")
if __name__=='__main__': main()
