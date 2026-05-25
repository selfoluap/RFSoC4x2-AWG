import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

interface PlotlyChartProps {
  title: string;
  x: number[];
  y: number[];
  xLabel: string;
  yLabel: string;
  tone?: "signal" | "spectrum";
}

export function PlotlyChart({
  title,
  x,
  y,
  xLabel,
  yLabel,
  tone = "signal",
}: PlotlyChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const styles = getComputedStyle(document.documentElement);
    const lineColorMap = {
      signal: styles.getPropertyValue("--plot-signal").trim() || "#38d27d",
      spectrum: styles.getPropertyValue("--plot-spectrum").trim() || "#f0b44d",
    };
    const axisColor = styles.getPropertyValue("--plot-axis").trim() || "rgba(255,255,255,0.12)";
    const gridColor = styles.getPropertyValue("--plot-grid").trim() || "rgba(255,255,255,0.08)";
    const textColor = styles.getPropertyValue("--text").trim() || "#f4efdf";
    const mutedColor = styles.getPropertyValue("--muted").trim() || "#a9b2a0";

    void Plotly.react(
      containerRef.current,
      [{ x, y, type: "scatter", mode: "lines", line: { color: lineColorMap[tone], width: 2 } }],
      {
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { family: '"IBM Plex Mono", monospace', color: textColor, size: 12 },
        title: {
          text: title,
          x: 0,
          xanchor: "left",
          font: { family: '"Fraunces", serif', size: 20, color: textColor },
        },
        margin: { t: 56, l: 54, r: 22, b: 54 },
        xaxis: {
          title: { text: xLabel, standoff: 12 },
          gridcolor: gridColor,
          zerolinecolor: axisColor,
          linecolor: axisColor,
          tickfont: { color: mutedColor },
        },
        yaxis: {
          title: { text: yLabel, standoff: 12 },
          gridcolor: gridColor,
          zerolinecolor: axisColor,
          linecolor: axisColor,
          tickfont: { color: mutedColor },
        },
      },
      { responsive: true, displaylogo: false }
    );
  }, [title, x, y, xLabel, yLabel, tone]);

  return <div ref={containerRef} className="plot" />;
}
