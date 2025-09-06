import json, platform, sys, time

def make_md_report(metrics: dict, extras: dict) -> str:
    lines = []
    lines.append("# Credit Scoring — Metrics Report\n")
    lines.append("## Summary\n")
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.4f}")
        else:
            lines.append(f"- **{k}**: {v}")
    lines.append("\n## Environment\n")
    py = sys.version.replace("\n", " ")
    lines.append(f"- Python: {py}")
    lines.append(f"- Platform: {platform.platform()}")
    for k, v in extras.items():
        lines.append(f"- {k}: {v}")
    lines.append("\n*Report generated:* " + time.strftime("%Y-%m-%d %H:%M:%S"))
    return "\n".join(lines)
