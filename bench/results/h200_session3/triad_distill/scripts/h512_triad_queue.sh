#!/usr/bin/env bash
# H=512 Triad-distill: from-scratch since H=512 incompat with H=384 init
# 3 seeds, soft Triad teacher, 160K steps each
# Auto-decision: if seed 0 < 2.45%, train seeds 1+2 + eval; else STOP
set -e
LOG=/workspace/h512_triad_queue.log
SUMMARY=/workspace/h512_triad_summary.txt
THRESHOLD_PCT=2.45
PF_TEACHERS='/workspace/persist/checkpoints/pathfinder_wide_long_d7/best_model.pt /workspace/persist/checkpoints/pathfinder_wide_long_d7_seed1/best_model.pt /workspace/persist/checkpoints/pathfinder_wide_long_d7_seed2/best_model.pt'

read_ler() {
  python3 -c "
import sys; sys.path.insert(0, '/workspace/pathfinder/train')
import torch
ck = torch.load('$1', weights_only=False, map_location='cpu')
print(ck.get('ler', 1.0))
" 2>&1
}

write_status() {
  printf '=== H=512 Triad-distill STATUS ===\nUpdate: %s\nPhase: %s\nDetails: %s\n' "$(date -u)" "$1" "$2" > $SUMMARY
}

train_h512() {
  local SEED=$1
  local CKPT=/workspace/persist/checkpoints/pathfinder_triad_h512_d7_seed${SEED}
  echo "=== H=512 seed=$SEED START $(date -u) ===" | tee -a $LOG
  python3 /workspace/train_seeded_wide_long_triad.py --seed $SEED \
    --distance 7 --hidden_dim 512 --steps 160000 \
    --batch 96 --noise_rate 0.007 \
    --muon_lr 0.002 --alpha_kl 0.7 --alpha_bce 0.3 \
    --ckpt $CKPT \
    --pf_teacher_ckpts $PF_TEACHERS \
    --eval_interval 10000 --log_interval 1000 \
    2>&1 | tee /workspace/h512_triad_seed${SEED}.log | tail -50 >> $LOG
  echo "=== H=512 seed=$SEED DONE $(date -u) ===" | tee -a $LOG
}

echo "=== H=512 Triad-distill queue START $(date -u) ===" | tee $LOG
write_status 'PHASE_1' 'H=512 Triad-distill seed 0 (from-scratch, ~5-6h)'

# Phase 1: seed 0
train_h512 0
SEED0_LER=$(read_ler /workspace/persist/checkpoints/pathfinder_triad_h512_d7_seed0/best_model.pt)
SEED0_PCT=$(python3 -c "print(float('$SEED0_LER') * 100)")
echo "H=512 seed 0 LER: $SEED0_LER (${SEED0_PCT}%)" | tee -a $LOG
write_status 'PHASE_1_DONE' "Seed 0 LER=${SEED0_PCT}% (threshold ${THRESHOLD_PCT}%)"

DECISION=$(python3 -c "print('GO' if float('$SEED0_LER') * 100 <  else 'STOP')")
if [[ "$DECISION" == 'GO' ]]; then
  echo 'GO: training seeds 1, 2' | tee -a $LOG
  write_status 'PHASE_2' 'Training H=512 seeds 1, 2 + eval'
  train_h512 1
  train_h512 2
  
  # Run full eval
  cat > /workspace/eval_triad_h512.py << 'PYEOF'
import sys, os, json
sys.path.insert(0, '/workspace'); sys.path.insert(0, '/workspace/pathfinder/train'); sys.path.insert(0, '/workspace/GNN_decoder')
import numpy as np, torch, pymatching
from ensemble_pf_lange import LangeWrapper, PathfinderMapper, wilson, make_circuit
from model import NeuralDecoder
device = torch.device('cuda')
PF_CKPTS = [f'/workspace/persist/checkpoints/pathfinder_triad_h512_d7_seed{s}/best_model.pt' for s in (0,1,2)]
def load(paths):
    models = []
    for p in paths:
        ck = torch.load(p, weights_only=False, map_location=device)
        m = NeuralDecoder(ck['config']).to(device); m.load_state_dict(ck['model_state_dict']); m.eval()
        models.append(m)
    return models
def pf_predict_avg(models, syn):
    with torch.no_grad():
        avg = None
        for m in models:
            lg = m(syn).cpu().numpy()
            avg = lg if avg is None else avg + lg
        return ((avg / len(models)) > 0).astype(np.uint8)
paths = [p for p in PF_CKPTS if os.path.exists(p)]
print(f'd=7: {len(paths)} H=512 PF model(s)', flush=True)
pf_models = load(paths)
NOISE_RATES = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015]
N_PER_SEED = 20000
SAMPLE_SEEDS = [3000, 3001, 3002, 3003, 3004]
d = 7
results = {}
for p in NOISE_RATES:
    c = make_circuit(d, p); pfm = PathfinderMapper(c)
    pf_e = la_e = pm_e = maj_e = all3 = tot = 0
    for sseed in SAMPLE_SEEDS:
        sampler = c.compile_detector_sampler(seed=sseed)
        det, obs = sampler.sample(shots=N_PER_SEED, separate_observables=True)
        det = det.astype(np.uint8); obs = obs.astype(np.uint8)
        dem = c.detector_error_model(decompose_errors=True)
        pm = pymatching.Matching.from_detector_error_model(dem)
        pm_pr = pm.decode_batch(det).astype(np.uint8)
        pf_pr = np.zeros_like(obs)
        for i in range(0, N_PER_SEED, 250):
            syn = pfm.to_tensor(det[i:i+250]).to(device)
            pf_pr[i:i+250] = pf_predict_avg(pf_models, syn)
        lw = LangeWrapper(d, d); lw.init_from_circuit(c)
        la_pr = np.zeros_like(obs)
        for i in range(0, N_PER_SEED, 500):
            la_pr[i:i+500] = lw.predict_batch(det[i:i+500])
        pfw = np.any(pf_pr != obs, axis=1); law = np.any(la_pr != obs, axis=1); pmw = np.any(pm_pr != obs, axis=1)
        maj = ((pf_pr.astype(int) + la_pr.astype(int) + pm_pr.astype(int)) >= 2).astype(np.uint8)
        majw = np.any(maj != obs, axis=1)
        pf_e += int(pfw.sum()); la_e += int(law.sum()); pm_e += int(pmw.sum())
        maj_e += int(majw.sum()); all3 += int((pfw & law & pmw).sum()); tot += N_PER_SEED
    pfl, pflo, pfhi = wilson(pf_e, tot); lal, lalo, lahi = wilson(la_e, tot)
    pml, pmlo, pmhi = wilson(pm_e, tot); mal, malo, mahi = wilson(maj_e, tot)
    non_pf = 'PF<<Lange' if pfhi < lalo else ('Lange<<PF' if lahi < pflo else 'overlap')
    non_pf_maj = 'PF<<Maj' if pfhi < malo else ('Maj<<PF' if mahi < pflo else 'overlap')
    print(f'  d={d} p={p}: PF={pfl*100:.4f}% [{pflo*100:.4f},{pfhi*100:.4f}]  Lange={lal*100:.4f}%  PM={pml*100:.4f}%  Maj={mal*100:.4f}% [{malo*100:.4f},{mahi*100:.4f}]  PFvsLange={non_pf}  PFvsMaj={non_pf_maj}', flush=True)
    results[f'd{d}_p{p}'] = {'d': d, 'p': p, 'n': tot, 'n_pf_seeds': len(pf_models),
        'pf_ler': pfl, 'pf_ci': [pflo, pfhi], 'lange_ler': lal, 'lange_ci': [lalo, lahi],
        'pm_ler': pml, 'pm_ci': [pmlo, pmhi], 'majority_ler': mal, 'majority_ci': [malo, mahi],
        'oracle_lb': all3 / tot}
os.makedirs('/workspace/persist/results', exist_ok=True)
with open('/workspace/persist/results/ensemble_triad_h512_d7.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved', flush=True)
PYEOF
  python3 /workspace/eval_triad_h512.py 2>&1 | tee /workspace/eval_triad_h512.log | tee -a $LOG
  
  python3 -c "
import json
with open('/workspace/persist/results/ensemble_triad_h512_d7.json') as f: r = json.load(f)
pf = r['d7_p0.007']
print('=== H=512 Triad-distill FINAL ===')
print(f\"d=7 p=0.007:  PF={pf['pf_ler']*100:.3f}%  Triad={pf['majority_ler']*100:.3f}%  Lange={pf['lange_ler']*100:.3f}%\")
print(f\"PF vs Triad: PF=[{pf['pf_ci'][0]*100:.3f},{pf['pf_ci'][1]*100:.3f}]  Triad=[{pf['majority_ci'][0]*100:.3f},{pf['majority_ci'][1]*100:.3f}]\")
import sys
pfhi, malo = pf['pf_ci'][1]*100, pf['majority_ci'][0]*100
verdict = 'STRICT WIN: PF beats Triad' if pfhi < malo else ('PF point estimate < Triad' if pf['pf_ler']<pf['majority_ler'] else 'Triad still beats PF')
print(f'Verdict at p=0.007: {verdict}')
" >> $SUMMARY
  cat $SUMMARY | tee -a $LOG
else
  printf '\nSeed 0 LER %s%% missed threshold %s%%; STOPPING.\n' $SEED0_PCT  >> $SUMMARY
  cat $SUMMARY | tee -a $LOG
fi

echo "=== H=512 queue EXIT $(date -u) ===" | tee -a $LOG
