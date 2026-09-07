"""Plot a solve JSON result: u, C, beta*f panels. Usage: plot_run.py <results.json> <out.png>"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print("usage: plot_run.py <results.json> <out.png>")
        sys.exit(2)
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, encoding="utf-8") as fh:
        data = json.load(fh)

    recs = data["records"]
    model = data["metadata"]["model"]["type"]
    beta = [r["beta"] for r in recs]
    u = [r["u"] for r in recs]
    c = [r["c"] for r in recs]
    bf = [r["beta"] * r["f"] for r in recs]
    has_exact = bool(recs) and recs[0]["exact"] is not None

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(beta, u, "o-", ms=3, label="iMPS")
    axes[0].set(xlabel=r"$\beta$", ylabel=r"$u$", title="energy density")
    axes[1].plot(beta, c, "o-", ms=3, label="iMPS")
    axes[1].set(xlabel=r"$\beta$", ylabel=r"$C$", title="specific heat")
    axes[2].plot(beta, bf, "o-", ms=3, label="iMPS")
    axes[2].set(xlabel=r"$\beta$", ylabel=r"$\beta f$", title="free energy")

    if has_exact:
        axes[0].plot(beta, [r["exact"]["u"] for r in recs], "k--", lw=1, label="exact")
        axes[1].plot(beta, [r["exact"]["c"] for r in recs], "k--", lw=1, label="exact")
        axes[2].plot(beta, [r["beta"] * r["exact"]["f"] for r in recs], "k--", lw=1, label="exact")

    if model in ("aklt", "aklt_projector"):
        ground = 0.0 if model == "aklt_projector" else -2.0 / 3.0
        axes[0].axhline(
            ground,
            color="r",
            ls=":",
            lw=1,
            label=rf"$u(T\to0)={ground:g}$",
        )
        axes[2].axhline(
            -math.log(3.0),
            color="r",
            ls=":",
            lw=1,
            label=r"$\beta f\ (T\to\infty)=-\ln 3$",
        )

    for ax in axes:
        ax.legend()
    fig.suptitle(f"{model} — iMPS purification finite-T")
    fig.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
