import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

interface ShapFeature {
  name: string;
  value: number;
}

interface ShapData {
  service: string;
  features: ShapFeature[];
  all_values?: Record<string, number>;
}

interface ShapExplainerProps {
  data: ShapData | null;
}

export default function ShapExplainer({ data }: ShapExplainerProps) {
  if (!data || data.features.length === 0) {
    return (
      <div className="panel shap-panel">
        <h3>SHAP Explanation</h3>
        <p className="text-muted">Waiting for anomaly detection to generate feature attributions...</p>
      </div>
    );
  }

  const sorted = [...data.features].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const chartData = sorted.slice(0, 8).map(f => ({
    name: f.name
      .replace(/_/g, ' ')
      .replace('percent', '%')
      .replace('transactions per minute', 'TPM'),
    value: f.value,
    absValue: Math.abs(f.value),
  }));

  return (
    <div className="panel shap-panel">
      <div className="panel-header">
        <h3>SHAP Feature Attribution</h3>
        <span className="shap-service">{data.service}</span>
      </div>
      <p className="shap-subtitle">
        Why the anomaly was flagged — red pushes toward anomaly, blue toward normal
      </p>
      <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 32)}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fontSize: 10, fill: '#94a3b8' }}
            axisLine={{ stroke: '#334155' }}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={140}
            tick={{ fontSize: 11, fill: '#e2e8f0' }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              background: '#1e293b',
              border: '1px solid #334155',
              borderRadius: 6,
              fontSize: 12,
            }}
            formatter={(val: number | undefined) => [(val ?? 0).toFixed(4), 'SHAP value']}
          />
          <ReferenceLine x={0} stroke="#475569" />
          <Bar dataKey="value" radius={[0, 3, 3, 0]} isAnimationActive={false}>
            {chartData.map((entry, i) => (
              <Cell key={i} fill={entry.value > 0 ? '#ef4444' : '#3b82f6'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
