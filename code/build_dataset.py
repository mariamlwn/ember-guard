"""Expand the curated Appendix A table (one line per test) into one row per observed
condition: (O2, flow, fuel) -> burning / extinction. Every row keeps its source + page.
Usage: python -I build_dataset.py ../data/bass2_appendixA_curated.psv ../data/bass2_dataset.csv"""
import sys, csv
SRC = "Olson et al. 2021, BASS-II Summary Report, NASA/TM-20210011385, Appendix A"
L_BURN, L_EXT = 0.5, 0.4   # below anemometer floor; Link et al. 2018: flows down to 0.4 cm/s, no spread below ~0.6
def flow(tok, ext):
    tok = tok.strip()
    if tok == "L": return (L_EXT if ext else L_BURN), "below_sensor_floor"
    if tok.startswith("~"): return float(tok[1:]), "estimated_from_fan_pot"
    return float(tok), "air_display"
lines = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip() and not l.startswith("#")]
rows = list(csv.DictReader(lines, delimiter="|"))
out = []
for r in rows:
    r = {k.strip(): (v or "").strip() for k, v in r.items()}
    oi = float(r["o2_cal_i"]); of = float(r["o2_cal_f"]) if r["o2_cal_f"] else oi
    if not (10 < of < 25): of = oi            # discard broken final readings (e.g. 0.6, 39.1)
    base = dict(test_id=r["test"], pi=r["pi"], material=r["material"], geometry=r["geometry"],
                thickness_mm=r["thk_mm"], width_cm=r["width_cm"], flow_config=r["config"],
                o2_pct=round((oi + of) / 2, 2), o2_initial=oi, o2_final=of,
                o2_flag="feb2014_calibration_reads_22.2" if oi == 22.2 else "",
                source=SRC + f", Table {r['table']}", page=r["page"], note=r["note"])
    seen = set()
    if r["burn_flows"] != "NA":
        for t in r["burn_flows"].split(","):
            v, q = flow(t, False)
            if (v, "burning") in seen: continue
            seen.add((v, "burning"))
            out.append({**base, "flow_cm_s": v, "flow_quality": q, "outcome": "burning", "mechanism": "sustained"})
    if r["end"] in ("EXT", "BLOW", "NOIGN"):
        v, q = flow(r["end_flow"], True)
        mech = {"EXT": "low_flow_or_oxygen_extinction", "BLOW": "blowoff", "NOIGN": "no_sustained_ignition"}[r["end"]]
        if r["end"] == "BLOW" and q == "air_display": pass
        out.append({**base, "flow_cm_s": v, "flow_quality": q, "outcome": "extinction", "mechanism": mech})
keys = ["row_id"] + list(out[0].keys())
with open(sys.argv[2], "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for i, o in enumerate(out, 1): w.writerow({"row_id": f"A{i:03d}", **o})
print(f"{len(rows)} tests -> {len(out)} rows")
