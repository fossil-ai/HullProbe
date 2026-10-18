"""Print the numbers and LaTeX rows for the GT-mask appendix from gt_mask_audit.json."""
import json, sys
from pathlib import Path

R = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else
                    "experiments/gt_mask_audit.json").read_text())
CELLS = ["hull-fwd", "hull-mid", "hull-aft", "sup-fwd", "sup-mid", "sup-aft"]
ARM_NAME = {"pred": "segmenter (both)", "gt": "annotation (both)", "gt_r": "annotation (recipient only)"}


def ci(d, k, f="{:.2f}"):
    v = d[k]
    return f"${f.format(v['value'])}$ {{\\scriptsize$[{f.format(v['ci'][0])},{f.format(v['ci'][1])}]$}}"


def sci(d, k):
    v = d[k]
    return (f"${v['value']:+.3f}$ {{\\scriptsize$[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]$}}")


print("skipped:", R["skipped"])
for sub in ("all_heldout", "test_split_only"):
    S = R[sub]
    print(f"\n% ---- {sub}: n={S['n']} ----")
    print("% seg agreement:", S["seg_agreement_mean"], "median disagree", S["seg_disagree_median"])
    print("% per-recipient profile r:", S["per_recipient_corr_profile_pearson"])
    for arm in ("pred", "gt", "gt_r"):
        a = S["arms"][arm]
        print(f"{ARM_NAME[arm]:28s} & {ci(a,'hull_share')} & ${a['hull_share_raw']['value']:.2f}$ & "
              f"${a['W1']['value']:.2f}$ & ${a['W_le2']['value']:.2f}$ & ${a['frac_r1']['value']:.2f}$ \\\\")
    for key in ("diff_gt_minus_pred", "diff_gt_r_minus_pred"):
        d = S[key]
        print(f"% {key}: hull-share {sci(d,'hull_share')}  raw {sci(d,'hull_share_raw')}  "
              f"W1 {sci(d,'W1')}  W<=2 {sci(d,'W_le2')}  r1 {sci(d,'frac_r1')}")
    print("% per-cell corrected influence (pred | gt | diff):")
    for c in CELLS:
        k = f"corr_{c}"
        print(f"{c:9s} & ${S['arms']['pred'][k]['value']:.3f}$ & ${S['arms']['gt'][k]['value']:.3f}$ & "
              f"{sci(S['diff_gt_minus_pred'], k)} \\\\")
