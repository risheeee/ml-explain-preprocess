import json
import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt

class ExplainReport:
    """Class for generating & formatting beginner friendly preprocessing reports."""

    def __init__(self):
        self.logs = {}      # reports dict (step)

    def add_step(self, step_name, step_report):
        """Add step's report."""
        self.logs[step_name] = step_report

    def generate(self, format = "json", output_dir = "reports"):
        """Generate report in JSON or Text with beginner friendly details"""

        if format == 'json':
            return json.dumps(self.logs, indent = 4)
        
        elif format == "text":
            text = "--------------- Preprocessing Report ---------------- \n"
            for step, report in self.logs.items():
                text += f"\nStep: {step.upper()}\n"
                text += f"Explanation: {report['explanation']}\n"
                text += f"Parameters used: {report['parameters']}\n"
                text += f"Impact: {report['impact']}\n"
                text += f"Statistics: \n"
                for key, value in report['stats'].items():
                    text += f"{key}: {value}\n"
                if "visuals" in report and report ['visuals']:
                    text += f"Visuals: Saved at {', '.join(report['visuals'])}\n"
                    if 'visual_descriptions' in report:
                        text += "Visual Descriptions: \n"
                        for desc in report['visual_descriptions']:
                            text += f" - {desc}\n"
                text += f"Tips for Beginners: {report['tips']}\n"
                text += "-" * 27
            return text
        else:
            raise ValueError("Unsupported format, please use 'json' or 'text'.")
        
    def save_plot(self, fig, filename, output_dir = "reports"):
        """Save plot to file & report base64 for embedding in the report"""
        os.makedirs(output_dir, exist_ok = True)
        path = os.path.join(output_dir, filename)
        fig.savefig(path)
        plt.close(fig)
        buf = BytesIO()     # return a base64 for embedding the plot in report.
        fig.savefig(buf, format = "png")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')