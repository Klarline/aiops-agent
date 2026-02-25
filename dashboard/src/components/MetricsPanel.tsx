import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface MetricsPanelProps {
  data: Record<string, Record<string, number>>;
  selectedService: string;
}

const METRIC_COLORS: Record<string, string> = {
  cpu_percent: '#ef4444',
  memory_percent: '#3b82f6',
  latency_p50_ms: '#f59e0b',
  error_rate: '#dc2626',
  request_rate: '#10b981',
  transactions_per_minute: '#8b5cf6',
};

export default function MetricsPanel({ data, selectedService }: MetricsPanelProps) {
  const serviceData = data[selectedService];
  if (!serviceData) return <div className="panel">No data for {selectedService}</div>;

  const metrics = Object.entries(serviceData).map(([key, value]) => ({
    name: key.replace(/_/g, ' '),
    value: typeof value === 'number' ? value.toFixed(2) : value,
    raw: value,
  }));

  return (
    <div className="panel">
      <h3>Metrics: {selectedService}</h3>
      <div className="metrics-grid">
        {metrics.map((m) => (
          <div key={m.name} className="metric-card">
            <div className="metric-label">{m.name}</div>
            <div className="metric-value">{m.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
