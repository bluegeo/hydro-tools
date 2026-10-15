"""Validate riparian zone delineation against stream geometry and reference maps.

Runs the `RiparianConnectivity` workflow and reports:

* Zone 2 (flood-prone) width along streams relative to bankfull width, binned by
  contributing area. In unconfined valleys the ratio should increase with area; in
  confined valleys it should remain near 1-2.
* Agreement (precision, recall, F1, IoU) between zones 1-2 and a reference floodplain
  mask, and between zone 3 (and the full region) and a reference wetland or riparian
  vegetation mask, when supplied.
* Optional sensitivity sweeps over `f2` and `h3`.

Example:

    python scripts/validate_riparian_zones.py dem.tif precip.tif out/ \\
        --floodplain-ref floodplain.tif --fringe-ref wetlands.tif \\
        --sweep-f2 2 3 4 --sweep-h3 1 2 3 5
"""

import argparse
import csv
import os
import shutil
from tempfile import TemporaryDirectory

import dask.array as da
import numpy as np
from scipy.stats import spearmanr

from hydrotools.morphology import RiparianConnectivity, region_width
from hydrotools.raster import Raster, from_raster, warp_like


def load(path):
    return np.ma.squeeze(from_raster(path).compute())


def reference_mask(path, template, tmp_dir):
    """Load a reference raster as a boolean mask aligned with the template."""
    if not Raster(path).matches(template):
        aligned = os.path.join(tmp_dir, os.path.basename(path))
        warp_like(path, aligned, template, resample_method="nearest", as_cog=False)
        path = aligned
    ref = load(path)
    return ~np.ma.getmaskarray(ref) & (np.ma.filled(ref, 0) != 0)


def agreement(predicted, reference):
    tp = float((predicted & reference).sum())
    fp = float((predicted & ~reference).sum())
    fn = float((~predicted & reference).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def zone_areas(zones, cell_area):
    return {f"zone{k}_km2": float((zones == k).sum() * cell_area / 1e6) for k in (1, 2, 3)}


def stream_metrics(rc, out_dir):
    """Zone 2 width / bankfull width along streams, written to CSV and summarized."""
    zones = load(rc.zones)
    floodplain = ~np.ma.getmaskarray(zones) & (zones <= 2)

    width_dst = os.path.join(out_dir, "zone2_width.tif")
    region_width(
        da.from_array(floodplain.reshape((1,) + floodplain.shape)), rc.streams, width_dst
    )

    width = load(width_dst)
    bfw = load(rc.bankfull_width)
    gradient = load(rc.stream_slope)
    area_km2 = load(rc.flow_accumulation) * rc.csx * rc.csy / 1e6

    on_stream = ~np.ma.getmaskarray(bfw) & ~np.ma.getmaskarray(width)
    rows, cols = np.where(on_stream)

    table_path = os.path.join(out_dir, "stream_metrics.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["row", "col", "area_km2", "bankfull_width_m", "gradient_pct", "zone2_width_m", "width_ratio"]
        )
        for r, c in zip(rows, cols):
            writer.writerow([
                r,
                c,
                f"{area_km2[r, c]:.4f}",
                f"{bfw[r, c]:.3f}",
                f"{np.ma.filled(gradient, np.nan)[r, c]:.3f}",
                f"{width[r, c]:.2f}",
                f"{width[r, c] / bfw[r, c]:.2f}",
            ])

    area = np.asarray(area_km2[on_stream])
    ratio = np.asarray(width[on_stream] / bfw[on_stream])
    rho, p = spearmanr(area, np.asarray(width[on_stream]))

    print("\nZone 2 width along streams")
    print(f"  stream cells: {len(area)}")
    print(f"  Spearman rho(area, zone 2 width) = {rho:.3f} (p = {p:.3g})")
    print("  width / bankfull width by contributing area:")
    edges = np.quantile(area, np.linspace(0, 1, 6))
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (area >= lo) & (area <= hi)
        if sel.any():
            print(
                f"    {lo:8.2f} - {hi:8.2f} km2  median {np.median(ratio[sel]):6.1f}"
                f"  IQR {np.quantile(ratio[sel], 0.25):6.1f} - {np.quantile(ratio[sel], 0.75):6.1f}"
            )
    print(f"  table: {table_path}")


def report_agreement(zones, floodplain_ref, fringe_ref):
    results = {}
    valid = ~np.ma.getmaskarray(zones)
    z = np.ma.filled(zones, 0)
    if floodplain_ref is not None:
        results["floodplain(zones 1-2)"] = agreement(valid & (z <= 2), floodplain_ref)
    if fringe_ref is not None:
        results["fringe(zone 3)"] = agreement(z == 3, fringe_ref)
        results["region(zones 1-3)"] = agreement(valid, fringe_ref)
    return results


def print_agreement(results):
    if not results:
        return
    print("\nAgreement with reference maps")
    print(f"  {'comparison':24s} {'precision':>9s} {'recall':>7s} {'f1':>6s} {'iou':>6s}")
    for name, m in results.items():
        print(
            f"  {name:24s} {m['precision']:9.3f} {m['recall']:7.3f} {m['f1']:6.3f} {m['iou']:6.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dem")
    parser.add_argument("precip", help="Mean annual precipitation (mm)")
    parser.add_argument("out_dir")
    parser.add_argument("--min-ws-area", type=float, default=1e6)
    parser.add_argument("--f1", type=float, default=1.0)
    parser.add_argument("--f2", type=float, default=3.0)
    parser.add_argument("--h3", type=float, default=3.0)
    parser.add_argument("--fringe-mode", choices=["rise", "convergence"], default="rise")
    parser.add_argument("--memory", type=int, default=4096)
    parser.add_argument("--floodplain-ref", help="Reference floodplain mask raster")
    parser.add_argument("--fringe-ref", help="Reference wetland / riparian vegetation mask raster")
    parser.add_argument("--sweep-f2", type=float, nargs="*", default=[])
    parser.add_argument("--sweep-h3", type=float, nargs="*", default=[])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rc = RiparianConnectivity(args.dem, args.precip)
    rc.extract_streams(min_ws_area=args.min_ws_area, memory=args.memory)
    rc.calc_stream_attributes()

    zone_kwargs = {"f1": args.f1, "fringe_mode": args.fringe_mode, "memory": args.memory}
    rc.calc_zones(f2=args.f2, h3=args.h3, **zone_kwargs)

    rc.calc_connectivity(os.path.join(args.out_dir, "zones.tif"), score_dst=os.path.join(args.out_dir, "score.tif"), vector=False)
    rc.calc_connectivity(os.path.join(args.out_dir, "zones.gpkg"))
    for name in ("streams", "bankfull_width", "stream_slope"):
        shutil.copy(getattr(rc, name), os.path.join(args.out_dir, f"{name}.tif"))

    cell_area = rc.csx * rc.csy
    zones = load(rc.zones)
    print("Zone areas (km2):", {k: round(v, 3) for k, v in zone_areas(np.ma.filled(zones, 0), cell_area).items()})

    stream_metrics(rc, args.out_dir)

    with TemporaryDirectory() as tmp_dir:
        floodplain_ref = reference_mask(args.floodplain_ref, args.dem, tmp_dir) if args.floodplain_ref else None
        fringe_ref = reference_mask(args.fringe_ref, args.dem, tmp_dir) if args.fringe_ref else None

        print_agreement(report_agreement(zones, floodplain_ref, fringe_ref))

        sweep = [("f2", v) for v in args.sweep_f2] + [("h3", v) for v in args.sweep_h3]
        if sweep:
            sweep_path = os.path.join(args.out_dir, "sensitivity.csv")
            with open(sweep_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["parameter", "value", "zone1_km2", "zone2_km2", "zone3_km2", "comparison", "precision", "recall", "f1", "iou"]
                writer.writerow(header)
                for param, value in sweep:
                    kwargs = dict(zone_kwargs, f2=args.f2, h3=args.h3)
                    kwargs[param] = value
                    rc.calc_zones(**kwargs)
                    z = load(rc.zones)
                    areas = zone_areas(np.ma.filled(z, 0), cell_area)
                    results = report_agreement(z, floodplain_ref, fringe_ref) or {"-": {}}
                    for name, m in results.items():
                        writer.writerow([
                            param,
                            value,
                            f"{areas['zone1_km2']:.4f}",
                            f"{areas['zone2_km2']:.4f}",
                            f"{areas['zone3_km2']:.4f}",
                            name,
                            *(f"{m.get(k, float('nan')):.4f}" for k in ("precision", "recall", "f1", "iou")),
                        ])
                    print(f"  {param} = {value}: " + ", ".join(f"{k} {v:.3f}" for k, v in areas.items()))
            print(f"\nSensitivity table: {sweep_path}")


if __name__ == "__main__":
    main()
