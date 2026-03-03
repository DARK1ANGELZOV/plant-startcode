from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ReportService:
    def build_summary(self, measurements: list[dict[str, Any]]) -> dict[str, Any]:
        if not measurements:
            return {'count': 0, 'by_class': {}, 'by_crop': {}}

        df = pd.DataFrame(measurements)
        by_class = (
            df.groupby('class_name')[['area_px', 'length_px']]
            .mean()
            .round(3)
            .to_dict(orient='index')
        )
        by_crop = (
            df.groupby('crop')[['area_px', 'length_px']]
            .mean()
            .round(3)
            .to_dict(orient='index')
        )
        mm_available = (
            ('area_mm2' in df.columns)
            and ('length_mm' in df.columns)
            and (df['area_mm2'].notna().any() or df['length_mm'].notna().any())
        )
        if mm_available:
            by_class_mm = (
                df.groupby('class_name')[['area_mm2', 'length_mm']]
                .mean()
                .round(3)
                .to_dict(orient='index')
            )
            by_crop_mm = (
                df.groupby('crop')[['area_mm2', 'length_mm']]
                .mean()
                .round(3)
                .to_dict(orient='index')
            )
        else:
            by_class_mm = {}
            by_crop_mm = {}
        return {
            'count': int(len(df)),
            'by_class': by_class,
            'by_crop': by_crop,
            'by_class_mm': by_class_mm,
            'by_crop_mm': by_crop_mm,
            'wheat_vs_arugula': self._compare_crops(df),
            'mm_metrics_available': bool(mm_available),
        }

    def _compare_crops(self, df: pd.DataFrame) -> dict[str, Any]:
        if 'crop' not in df.columns:
            return {}

        crops = set(df['crop'].astype(str).unique())
        if not {'Wheat', 'Arugula'}.issubset(crops):
            return {'note': 'Сравнение Wheat vs Arugula доступно при наличии обеих культур в батче.'}

        wheat = df[df['crop'] == 'Wheat']
        arugula = df[df['crop'] == 'Arugula']

        if (
            ('length_mm' in df.columns)
            and ('area_mm2' in df.columns)
            and (df['length_mm'].notna().any() and df['area_mm2'].notna().any())
        ):
            return {
                'length_px_delta': float(wheat['length_px'].mean() - arugula['length_px'].mean()),
                'area_px_delta': float(wheat['area_px'].mean() - arugula['area_px'].mean()),
                'length_mm_delta': float(wheat['length_mm'].mean() - arugula['length_mm'].mean()),
                'area_mm2_delta': float(wheat['area_mm2'].mean() - arugula['area_mm2'].mean()),
            }
        return {
            'length_px_delta': float(wheat['length_px'].mean() - arugula['length_px'].mean()),
            'area_px_delta': float(wheat['area_px'].mean() - arugula['area_px'].mean()),
            'note': 'mm-сравнение недоступно без валидной калибровки.',
        }

    def save_distribution_plot(self, measurements: list[dict[str, Any]], out_path: Path) -> None:
        if not measurements:
            return

        df = pd.DataFrame(measurements)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
        mm_available = (
            ('area_mm2' in df.columns)
            and ('length_mm' in df.columns)
            and (df['area_mm2'].notna().any() and df['length_mm'].notna().any())
        )

        if mm_available:
            df.boxplot(column='area_mm2', by='class_name', ax=axes[0])
            axes[0].set_title('Area Distribution by Class')
            axes[0].set_ylabel('mm^2')

            df.boxplot(column='length_mm', by='class_name', ax=axes[1])
            axes[1].set_title('Length Distribution by Class')
            axes[1].set_ylabel('mm')
        else:
            df.boxplot(column='area_px', by='class_name', ax=axes[0])
            axes[0].set_title('Area Distribution by Class')
            axes[0].set_ylabel('px^2')

            df.boxplot(column='length_px', by='class_name', ax=axes[1])
            axes[1].set_title('Length Distribution by Class')
            axes[1].set_ylabel('px')

        plt.suptitle('')
        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    def save_pdf_report(
        self,
        out_path: Path,
        run_id: str,
        summary: dict[str, Any],
        recommendations: list[dict[str, str]],
    ) -> None:
        c = canvas.Canvas(str(out_path), pagesize=A4)
        width, height = A4

        y = height - 50
        c.setFont('Helvetica-Bold', 14)
        c.drawString(40, y, f'Agro AI Report - {run_id}')

        y -= 30
        c.setFont('Helvetica', 10)
        c.drawString(40, y, f"Detected instances: {summary.get('count', 0)}")

        y -= 25
        c.setFont('Helvetica-Bold', 11)
        c.drawString(40, y, 'Class Aggregates (mean):')
        y -= 18

        c.setFont('Helvetica', 9)
        mm_available = bool(summary.get('mm_metrics_available', False))
        for class_name, values in summary.get('by_class', {}).items():
            if mm_available:
                mm_values = summary.get('by_class_mm', {}).get(class_name, {})
                line = (
                    f"{class_name}: area={values.get('area_px', 0)} px2, length={values.get('length_px', 0)} px, "
                    f"area_mm2={mm_values.get('area_mm2', 0)} mm2, length_mm={mm_values.get('length_mm', 0)} mm"
                )
            else:
                line = f"{class_name}: area={values.get('area_px', 0)} px2, length={values.get('length_px', 0)} px"
            c.drawString(40, y, line)
            y -= 14

        y -= 10
        c.setFont('Helvetica-Bold', 11)
        c.drawString(40, y, 'Recommendations:')
        y -= 18

        c.setFont('Helvetica', 9)
        if not recommendations:
            c.drawString(40, y, 'No recommendations generated.')
            y -= 14
        else:
            for rec in recommendations:
                msg = f"[{rec['severity']}] {rec['message']} | Action: {rec['action']}"
                c.drawString(40, y, msg[:110])
                y -= 14
                if y < 70:
                    c.showPage()
                    y = height - 50

        c.save()
