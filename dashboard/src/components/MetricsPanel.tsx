import { useMemo, useState } from 'react';
import {
  Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Area, ComposedChart,
} from 'recharts';

interface StepData {
  step: number;
  metrics: Record<string, Record<string, number>>;
  anomaly_scores: Record<string, number>;
}

interface MetricsPanelProps {
  timeline: StepData[];
  selectedService: string;
  allServices: string[];
  detectionStep: number;
  latestMetrics: Record<string, Record<string, number>>;
}

const METRIC_GROUPS = [
  {
    label: 'Resource Usage',
    metrics: ['cpu_percent', 'memory_percent'],
    unit: '%',
    domain: [0, 100] as [number, number],
  },
  {
    label: 'Latency',
    metrics: ['latency_p50_ms', 'latency_p99_ms'],
    unit: 'ms',
  },
  {
    label: 'Throughput',
    metrics: ['request_rate', 'transactions_per_minute'],
    unit: '/min',
  },
  {
    label: 'Errors & I/O',
    metrics: ['error_rate', 'disk_io_percent'],
    unit: '',
  },
];

const METRIC_COLORS: Record<string, string> = {
  cpu_percent: '#ef4444',
  memory_percent: '#3b82f6',
  latency_p50_ms: '#f59e0b',
  latency_p99_ms: '#f97316',
  error_rate: '#dc2626',
  request_rate: '#10b981',
  disk_io_percent: '#6366f1',
  transactions_per_minute: '#8b5cf6',
};

const METRIC_LABELS: Record<string, string> = {
  cpu_percent: 'CPU %',
  memory_percent: 'Memory %',
  latency_p50_ms: 'Latency P50',
  latency_p99_ms: 'Latency P99',
  error_rate: 'Error Rate',
  request_rate: 'Req/s',
  disk_io_percent: 'Disk I/O %',
  transactions_per_minute: 'TPM',
};

function AnomalyScoreBar({ scores }: { scores: Record<string, number> }) {
  if (!scores || Object.keys(scores).length === 0) return null;

  return (
    <div className="anomaly-bar">
      <div className="anomaly-bar-label">Anomaly Scores</div>
      <div className="anomaly-bar-items">
        {Object.entries(scores).map(([svc, score]) => (
          <div key={svc} className="anomaly-bar-item">
            <span className="anomaly-svc">{svc.split('-')[0]}</span>
            <div className="anomaly-bar-track">
              <div
                className={`anomaly-bar-fill ${score > 0.5 ? 'anomalous' : score > 0.3 ? 'warning' : ''}`}
                style={{ width: `${Math.min(score * 100, 100)}%` }}
              />
            </div>
            <span className={`anomaly-score ${score > 0.5 ? 'anomalous' : ''}`}>
              {score.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function MetricCard({ label, value, unit }: { label: string; value: number; unit: string }) {
  return (
    <div className="metric-card-small">
      <div className="metric-card-label">{label}</div>
      <div className="metric-card-value">
        {typeof value === 'number' ? value.toFixed(1) : '—'}
        <span className="metric-card-unit">{unit}</span>
      </div>
    </div>
  );
}

export default function MetricsPanel({
  timeline,
  selectedService,
  detectionStep,
  latestMetrics,
}: MetricsPanelProps) {
  const [viewMode, setViewMode] = useState<'charts' | 'overview'>('charts');

  const chartData = useMemo(() => {
    return timeline.map(step => {
      const svcMetrics = step.metrics[selectedService] || {};
      return {
        step: step.step,
        ...svcMetrics,
        _anomalyScore: step.anomaly_scores?.[selectedService] || 0,
      };
    });
  }, [timeline, selectedService]);

  const lastScores = timeline.length > 0
    ? timeline[timeline.length - 1].anomaly_scores || {}
    : {};

  const currentMetrics = latestMetrics[selectedService] || {};

  if (timeline.length === 0) {
    return (
      <div className="panel metrics-panel">
        <div className="panel-header">
          <h3>Service Metrics</h3>
        </div>
        <div className="metrics-empty">
          <div className="metrics-empty-icon">{'📊'}</div>
          <p>Run a scenario to see live metrics</p>
        </div>
      </div>
    );
  }

  return (
    <div className="panel metrics-panel">
      <div className="panel-header">
        <h3>Metrics: {selectedService}</h3>
        <div className="view-toggle">
          <button className={viewMode === 'charts' ? 'active' : ''} onClick={() => setViewMode('charts')}>
            Charts
          </button>
          <button className={viewMode === 'overview' ? 'active' : ''} onClick={() => setViewMode('overview')}>
            Overview
          </button>
        </div>
      </div>

      <AnomalyScoreBar scores={lastScores} />

      {viewMode === 'overview' ? (
        <div className="metrics-overview-grid">
          {Object.entries(currentMetrics).map(([key, val]) => (
            <MetricCard
              key={key}
              label={METRIC_LABELS[key] || key}
              value={val}
              unit={key.includes('percent') || key === 'error_rate' ? '%' : key.includes('ms') ? 'ms' : ''}
            />
          ))}
        </div>
      ) : (
        <div className="charts-grid">
          {METRIC_GROUPS.map(group => (
            <div key={group.label} className="chart-panel">
              <div className="chart-title">{group.label}</div>
              <ResponsiveContainer width="100%" height={140}>
                <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="step"
                    tick={{ fontSize: 10, fill: '#64748b' }}
                    tickLine={false}
                    axisLine={{ stroke: '#334155' }}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: '#64748b' }}
                    tickLine={false}
                    axisLine={false}
                    domain={group.domain || ['auto', 'auto']}
                  />
                  <Tooltip
                    contentStyle={{
                      background: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      fontSize: 12,
                    }}
                    labelStyle={{ color: '#94a3b8' }}
                  />
                  {detectionStep > 0 && (
                    <ReferenceLine
                      x={detectionStep}
                      stroke="#ef4444"
                      strokeDasharray="4 2"
                      label={{
                        value: 'DETECTED',
                        fill: '#ef4444',
                        fontSize: 9,
                        position: 'top',
                      }}
                    />
                  )}
                  {group.metrics.map(m => (
                    <Line
                      key={m}
                      type="monotone"
                      dataKey={m}
                      stroke={METRIC_COLORS[m]}
                      strokeWidth={1.5}
                      dot={false}
                      name={METRIC_LABELS[m] || m}
                      isAnimationActive={false}
                    />
                  ))}
                  {group.metrics.some(m => m === 'error_rate') && (
                    <Area
                      dataKey="_anomalyScore"
                      fill="#ef444420"
                      stroke="none"
                      name="Anomaly"
                      isAnimationActive={false}
                    />
                  )}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
