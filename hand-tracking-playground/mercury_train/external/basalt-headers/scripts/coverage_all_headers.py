#!/usr/bin/env python3

import argparse
import json
import pathlib
import subprocess
import sys


def run_llvm_cov_export(llvm_cov, profdata, bins):
    cmd = [llvm_cov, "export", "-summary-only", f"-instr-profile={profdata}"]
    for b in bins:
        cmd.extend(["-object", b])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"llvm-cov export failed with code {proc.returncode}")
    return json.loads(proc.stdout)


def summary_for_file(file_entry):
    s = file_entry.get("summary", {})
    lines = s.get("lines", {})
    funcs = s.get("functions", {})
    regions = s.get("regions", {})
    branches = s.get("branches", {})
    return {
        "lines_count": int(lines.get("count", 0)),
        "lines_covered": int(lines.get("covered", 0)),
        "functions_count": int(funcs.get("count", 0)),
        "functions_covered": int(funcs.get("covered", 0)),
        "regions_count": int(regions.get("count", 0)),
        "regions_covered": int(regions.get("covered", 0)),
        "branches_count": int(branches.get("count", 0)),
        "branches_covered": int(branches.get("covered", 0)),
    }


def pct(covered, total):
    if total == 0:
        return "0.00%"
    return f"{(100.0 * covered / total):.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Coverage summary for all basalt headers.")
    parser.add_argument("--bins-file", required=True)
    parser.add_argument("--profdata", required=True)
    parser.add_argument("--headers-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--llvm-cov", default="llvm-cov")
    args = parser.parse_args()

    repo_root = pathlib.Path.cwd().resolve()
    headers_root = (repo_root / args.headers_root).resolve()
    all_headers = sorted(
        [*headers_root.rglob("*.h"), *headers_root.rglob("*.hpp")],
        key=lambda p: str(p),
    )
    if not all_headers:
        raise RuntimeError(f"No headers found under {headers_root}")

    bins = []
    with open(args.bins_file, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                bins.append(p)
    if not bins:
        raise RuntimeError("No coverage binaries found.")

    files = {}
    export_json = run_llvm_cov_export(args.llvm_cov, args.profdata, bins)
    for data_entry in export_json.get("data", []):
        for file_entry in data_entry.get("files", []):
            filename = pathlib.Path(file_entry["filename"]).resolve()
            cur = summary_for_file(file_entry)
            key = str(filename)
            if key not in files:
                files[key] = cur
            else:
                # This should not happen with a single llvm-cov call for a given file
                # but we keep a simple max merge just in case.
                prev = files[key]
                for k in cur:
                    files[key][k] = max(prev[k], cur[k])

    rows = []
    totals = {
        "lines_count": 0,
        "lines_covered": 0,
        "functions_count": 0,
        "functions_covered": 0,
        "regions_count": 0,
        "regions_covered": 0,
        "branches_count": 0,
        "branches_covered": 0,
    }

    for hdr in all_headers:
        st = files.get(str(hdr), None)
        if st is None:
            st = {
                "lines_count": 0,
                "lines_covered": 0,
                "functions_count": 0,
                "functions_covered": 0,
                "regions_count": 0,
                "regions_covered": 0,
                "branches_count": 0,
                "branches_covered": 0,
            }
        for k in totals:
            totals[k] += st[k]
        rel = hdr.relative_to(repo_root)
        rows.append((str(rel), st))

    out_lines = []
    out_lines.append("Coverage Summary For All Headers (include/basalt)")
    out_lines.append(
        "file | line_cov | func_cov | region_cov | branch_cov | lines | funcs | regions | branches"
    )
    out_lines.append("---|---:|---:|---:|---:|---:|---:|---:|---:")
    for rel, st in rows:
        out_lines.append(
            f"{rel} | "
            f"{pct(st['lines_covered'], st['lines_count'])} | "
            f"{pct(st['functions_covered'], st['functions_count'])} | "
            f"{pct(st['regions_covered'], st['regions_count'])} | "
            f"{pct(st['branches_covered'], st['branches_count'])} | "
            f"{st['lines_covered']}/{st['lines_count']} | "
            f"{st['functions_covered']}/{st['functions_count']} | "
            f"{st['regions_covered']}/{st['regions_count']} | "
            f"{st['branches_covered']}/{st['branches_count']}"
        )

    out_lines.append("")
    out_lines.append(
        "TOTAL | "
        f"{pct(totals['lines_covered'], totals['lines_count'])} | "
        f"{pct(totals['functions_covered'], totals['functions_count'])} | "
        f"{pct(totals['regions_covered'], totals['regions_count'])} | "
        f"{pct(totals['branches_covered'], totals['branches_count'])} | "
        f"{totals['lines_covered']}/{totals['lines_count']} | "
        f"{totals['functions_covered']}/{totals['functions_count']} | "
        f"{totals['regions_covered']}/{totals['regions_count']} | "
        f"{totals['branches_covered']}/{totals['branches_count']}"
    )

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
