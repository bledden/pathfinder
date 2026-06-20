"""Extract the exact measurement->detector incidence from Stim's m2d converter by probing
with unit inputs (measurement j = 1, rest 0) and seeing which detectors flip. Verifiable,
no fragile circuit-text parsing."""
import numpy as np, stim

def detector_to_measurements(distance, rounds):
    circ = stim.Circuit.generated('surface_code:rotated_memory_z', distance=distance, rounds=rounds)
    nm = circ.num_measurements
    m2d = circ.compile_m2d_converter()
    # baseline: all-zero measurements -> detectors (should be all zero)
    base,_ = m2d.convert(measurements=np.zeros((1,nm),dtype=bool), separate_observables=True)
    base = base[0]
    nd = base.shape[0]
    det_to_meas = [[] for _ in range(nd)]
    for j in range(nm):
        x = np.zeros((1,nm),dtype=bool); x[0,j]=True
        d,_ = m2d.convert(measurements=x, separate_observables=True)
        flipped = np.where(d[0] ^ base)[0]
        for di in flipped:
            det_to_meas[di].append(j)
    return det_to_meas, nm, nd, circ

if __name__=="__main__":
    import sys
    D,R=int(sys.argv[1]),int(sys.argv[2])
    dtm,nm,nd,_=detector_to_measurements(D,R)
    sizes=[len(x) for x in dtm]
    open('/tmp/detmap.txt','w').write(
      f"d={D} r={R}: n_meas={nm} n_det={nd}\n"
      f"detector sizes (how many measurements XOR per detector): min={min(sizes)} max={max(sizes)} "
      f"counts={dict(zip(*np.unique(sizes,return_counts=True)))}\n")
    print(open('/tmp/detmap.txt').read())
