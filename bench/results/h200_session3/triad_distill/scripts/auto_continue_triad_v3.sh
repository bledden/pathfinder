#!/usr/bin/env bash
# Watcher v3: 3-branch tree (soft → hardlabel → warm-init from existing PFWL3S)
LOG=/workspace/auto_continue_triad.log
SUMMARY=/workspace/triad_experiment_summary.txt
THRESHOLD=0.0265
PF_TEACHERS='/workspace/persist/checkpoints/pathfinder_wide_long_d7/best_model.pt /workspace/persist/checkpoints/pathfinder_wide_long_d7_seed1/best_model.pt /workspace/persist/checkpoints/pathfinder_wide_long_d7_seed2/best_model.pt'

read_ler() {
  python3 -c "
import sys; sys.path.insert(0, '/workspace/pathfinder/train')
import torch
ck = torch.load('$1', weights_only=False, map_location='cpu')
print(ck.get('ler', 1.0))
" 2>&1
}

decide() {
  python3 -c "x = float('$1'); print('GO' if x <=  else 'PIVOT')" 2>&1
}

write_status() {
  echo '=== Triad-distillation experiment STATUS ===' > $SUMMARY
  echo "Last update: $(date -u)" >> $SUMMARY
  echo "Current phase: $1" >> $SUMMARY
  echo "Details: $2" >> $SUMMARY
}

train_soft_continuation() {
  for SEED in 1 2; do
    python3 /workspace/train_seeded_wide_long_triad.py --seed $SEED \
      --distance 7 --hidden_dim 384 --steps 160000 \
      --batch 128 --noise_rate 0.007 \
      --alpha_kl 0.7 --alpha_bce 0.3 \
      --ckpt /workspace/persist/checkpoints/pathfinder_triad_distill_d7_seed${SEED} \
      --pf_teacher_ckpts $PF_TEACHERS \
      --eval_interval 10000 --log_interval 1000 \
      2>&1 | tee /workspace/triad_distill_seed${SEED}.log | tail -50 >> $LOG
  done
  python3 /workspace/eval_triad_distill.py 2>&1 | tee /workspace/eval_triad_distill.log | tee -a $LOG
}

train_hardlabel_continuation() {
  for SEED in 1 2; do
    python3 /workspace/train_seeded_triad_hardlabel.py --seed $SEED \
      --distance 7 --hidden_dim 384 --steps 160000 \
      --batch 128 --noise_rate 0.007 \
      --alpha_true 0.3 --alpha_triad 0.7 \
      --ckpt /workspace/persist/checkpoints/pathfinder_triad_hardlabel_d7_seed${SEED} \
      --pf_teacher_ckpts $PF_TEACHERS \
      --eval_interval 10000 --log_interval 1000 \
      2>&1 | tee /workspace/triad_hardlabel_seed${SEED}.log | tail -50 >> $LOG
  done
}

echo "=== Watcher v3 START $(date -u) ===" | tee $LOG

# Phase 1: wait for soft seed 0 (already running)
write_status 'PHASE_1' 'Waiting for soft Triad seed 0'
while pgrep -f triad_distill_test.sh > /dev/null; do
  sleep 60
done
SOFT_LER=$(read_ler /workspace/persist/checkpoints/pathfinder_triad_distill_d7_seed0/best_model.pt)
echo "Soft seed 0 LER: $SOFT_LER" | tee -a $LOG
write_status 'PHASE_1_DONE' "Soft Triad seed 0 LER=$SOFT_LER (threshold )"

if [[ "$(decide $SOFT_LER)" == 'GO' ]]; then
  echo 'Soft GO: training soft seeds 1, 2' | tee -a $LOG
  write_status 'PHASE_1_HAPPY' 'Training soft seeds 1, 2 + eval'
  train_soft_continuation
  python3 /workspace/_write_done_summary.py > $SUMMARY
  cat $SUMMARY | tee -a $LOG
  exit 0
fi

# Phase 2: hardlabel pivot
echo 'PIVOT: hardlabel Triad seed 0' | tee -a $LOG
write_status 'PHASE_2' "Soft missed ($SOFT_LER); pivoting to hardlabel"
cat > /workspace/train_seeded_triad_hardlabel.py << 'PYEOF'
import sys, argparse, torch, numpy as np, random
p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, required=True)
args, rest = p.parse_known_args()
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed); random.seed(args.seed)
sys.argv = [sys.argv[0]] + rest
exec(open('/workspace/train_distill_triad_hardlabel.py').read())
PYEOF

python3 /workspace/train_seeded_triad_hardlabel.py --seed 0 \
  --distance 7 --hidden_dim 384 --steps 160000 \
  --batch 128 --noise_rate 0.007 \
  --alpha_true 0.3 --alpha_triad 0.7 \
  --ckpt /workspace/persist/checkpoints/pathfinder_triad_hardlabel_d7_seed0 \
  --pf_teacher_ckpts $PF_TEACHERS \
  --eval_interval 10000 --log_interval 1000 \
  2>&1 | tee /workspace/triad_hardlabel_seed0.log | tail -50 >> $LOG
HARDLABEL_LER=$(read_ler /workspace/persist/checkpoints/pathfinder_triad_hardlabel_d7_seed0/best_model.pt)
echo "Hardlabel seed 0 LER: $HARDLABEL_LER" | tee -a $LOG
write_status 'PHASE_2_DONE' "Hardlabel seed 0 LER=$HARDLABEL_LER"

if [[ "$(decide $HARDLABEL_LER)" == 'GO' ]]; then
  echo 'Hardlabel GO: training seeds 1, 2' | tee -a $LOG
  write_status 'PHASE_2_HAPPY' 'Training hardlabel seeds 1, 2 + eval'
  train_hardlabel_continuation
fi

# Phase 3: warm-init from existing PFWL3S + soft Triad teacher (most likely to work)
echo 'PHASE 3: warm-init from existing PFWL3S, soft Triad teacher' | tee -a $LOG
write_status 'PHASE_3' "Hardlabel also missed ($HARDLABEL_LER); pivoting to warm-init"

python3 /workspace/train_seeded_wide_long_triad.py --seed 0 \
  --distance 7 --hidden_dim 384 --steps 80000 \
  --batch 128 --noise_rate 0.007 \
  --alpha_kl 0.7 --alpha_bce 0.3 \
  --init /workspace/persist/checkpoints/pathfinder_wide_long_d7/best_model.pt \
  --ckpt /workspace/persist/checkpoints/pathfinder_triad_warminit_d7_seed0 \
  --pf_teacher_ckpts $PF_TEACHERS \
  --eval_interval 5000 --log_interval 1000 \
  2>&1 | tee /workspace/triad_warminit_seed0.log | tail -50 >> $LOG
WARMINIT_LER=$(read_ler /workspace/persist/checkpoints/pathfinder_triad_warminit_d7_seed0/best_model.pt)
echo "Warm-init seed 0 LER: $WARMINIT_LER" | tee -a $LOG
write_status 'PHASE_3_DONE' "Warm-init seed 0 LER=$WARMINIT_LER"

if [[ "$(decide $WARMINIT_LER)" == 'GO' ]]; then
  echo 'Warm-init GO: training warm-init seeds 1, 2' | tee -a $LOG
  write_status 'PHASE_3_HAPPY' 'Training warm-init seeds 1, 2 + eval'
  for SEED in 1 2; do
    python3 /workspace/train_seeded_wide_long_triad.py --seed $SEED \
      --distance 7 --hidden_dim 384 --steps 80000 \
      --batch 128 --noise_rate 0.007 \
      --alpha_kl 0.7 --alpha_bce 0.3 \
      --init /workspace/persist/checkpoints/pathfinder_wide_long_d7_seed${SEED}/best_model.pt \
      --ckpt /workspace/persist/checkpoints/pathfinder_triad_warminit_d7_seed${SEED} \
      --pf_teacher_ckpts $PF_TEACHERS \
      --eval_interval 5000 --log_interval 1000 \
      2>&1 | tee /workspace/triad_warminit_seed${SEED}.log | tail -50 >> $LOG
  done
fi

# Final summary
cat > $SUMMARY << SEOF
=== Triad-distillation experiment FINAL SUMMARY (v3 watcher, 3 branches) ===
Generated: $(date -u)

Branch 1 (soft Triad from-scratch, seed 0): LER = $SOFT_LER (decision: $(decide $SOFT_LER))
Branch 2 (hardlabel from-scratch, seed 0): LER = ${HARDLABEL_LER:-N/A} (decision: $(decide ${HARDLABEL_LER:-1.0}))
Branch 3 (soft Triad warm-init from PFWL3S, seed 0): LER = ${WARMINIT_LER:-N/A} (decision: $(decide ${WARMINIT_LER:-1.0}))

Threshold for 'GO seeds 1+2':  ('GO' = train more seeds, 'PIVOT' = move to next branch)
Goal: PF individual LER <= 2.38% (strictly beat the original Triad).

Eval results (only present if branch went happy path):
  Soft Triad: /workspace/persist/results/ensemble_triad_distill_d7.json
  Hardlabel: /workspace/persist/results/ensemble_triad_hardlabel_d7.json (if hardlabel happy)
  Warm-init: TBD (no full eval set up yet for warm-init branch)

Best ckpts (sorted by approach):
  Soft from-scratch: /workspace/persist/checkpoints/pathfinder_triad_distill_d7_seed{0,1,2}/
  Hardlabel from-scratch: /workspace/persist/checkpoints/pathfinder_triad_hardlabel_d7_seed{0,1,2}/
  Warm-init: /workspace/persist/checkpoints/pathfinder_triad_warminit_d7_seed{0,1,2}/

Logs: /workspace/auto_continue_triad.log
SEOF
cat $SUMMARY | tee -a $LOG
echo "=== Watcher v3 EXIT $(date -u) ===" | tee -a $LOG
