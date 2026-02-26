import { useState, useCallback } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface CompareResult {
  scenario: string;
  ml_agent: AgentResult;
  baseline: AgentResult;
}

interface AgentResult {
  name: string;
  detection_step: number;
  mttr_seconds: number | null;
  total_steps: number;
  evaluation: {
    detection: boolean;
    localization: boolean;
    diagnosis: boolean;
    mitigation: boolean;
    score: number;
  };
  log?: any[];
  shap?: any;
}

interface Props {
  scenarios: string[];
  apiBase: string;
}

function TaskGrid({ eval: ev }: { eval: AgentResult['evaluation'] }) {
  const tasks = [
    { label: 'Detect', pass: ev.detection },
    { label: 'Locate', pass: ev.localization },
    { label: 'Diagnose', pass: ev.diagnosis },
    { label: 'Mitigate', pass: ev.mitigation },
  ];
  return (
    <div className="compare-tasks">
      {tasks.map(t => (
        <span key={t.label} className={`compare-task ${t.pass ? 'pass' : 'fail'}`}>
          {t.pass ? '✓' : '✗'} {t.label}
        </span>
      ))}
    </div>
  );
}

export default function IncidentComparison({ scenarios, apiBase }: Props) {
  const [selected, setSelected] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CompareResult | null>(null);

  const runComparison = useCallback(async () => {
    if (!selected) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(`${apiBase}/agent/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario_id: selected }),
      });
      const data = await res.json();
      setResult(data);
    } catch (e: any) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [selected, apiBase]);

  const mttrData = result ? [
    {
      name: result.ml_agent.name,
      mttr: result.ml_agent.mttr_seconds || 0,
      fill: '#22c55e',
    },
    {
      name: result.baseline.name,
      mttr: result.baseline.mttr_seconds || result.baseline.total_steps * 10,
      fill: '#ef4444',
    },
  ] : [];

  const scoreData = result ? [
    {
      task: 'Detection',
      ml: result.ml_agent.evaluation.detection ? 1 : 0,
      baseline: result.baseline.evaluation.detection ? 1 : 0,
    },
    {
      task: 'Localization',
      ml: result.ml_agent.evaluation.localization ? 1 : 0,
      baseline: result.baseline.evaluation.localization ? 1 : 0,
    },
    {
      task: 'Diagnosis',
      ml: result.ml_agent.evaluation.diagnosis ? 1 : 0,
      baseline: result.baseline.evaluation.diagnosis ? 1 : 0,
    },
    {
      task: 'Mitigation',
      ml: result.ml_agent.evaluation.mitigation ? 1 : 0,
      baseline: result.baseline.evaluation.mitigation ? 1 : 0,
    },
  ] : [];

  return (
    <div className="compare-container">
      <div className="compare-header">
        <h2>Incident Comparison</h2>
        <p className="text-muted">Run the same fault scenario with ML Agent vs Static Threshold baseline</p>
      </div>

      <div className="compare-controls">
        <select value={selected} onChange={e => setSelected(e.target.value)} className="scenario-select">
          <option value="">Select scenario...</option>
          {scenarios.map(s => (
            <option key={s} value={s}>{s.replace(/_/g, ' ')}</option>
          ))}
        </select>
        <button className="btn btn-primary" onClick={runComparison} disabled={!selected || loading}>
          {loading ? 'Running...' : 'Compare Agents'}
        </button>
      </div>

      {loading && (
        <div className="compare-loading">
          <div className="spinner" />
          <p>Running both agents on the same scenario...</p>
        </div>
      )}

      {result && (
        <div className="compare-results">
          <div className="compare-cards">
            <div className="compare-card ml">
              <div className="compare-card-header">
                <span className="compare-card-badge ml">ML Agent</span>
                <span className="compare-card-score">
                  {(result.ml_agent.evaluation.score * 100).toFixed(0)}%
                </span>
              </div>
              <TaskGrid eval={result.ml_agent.evaluation} />
              <div className="compare-stat">
                <span className="compare-stat-label">Time to Remediate</span>
                <span className="compare-stat-value">
                  {result.ml_agent.mttr_seconds ? `${result.ml_agent.mttr_seconds}s` : 'Did not resolve'}
                </span>
              </div>
              <div className="compare-stat">
                <span className="compare-stat-label">Detection Step</span>
                <span className="compare-stat-value">
                  {result.ml_agent.detection_step > 0 ? `Step ${result.ml_agent.detection_step}` : 'Not detected'}
                </span>
              </div>
            </div>

            <div className="compare-vs">VS</div>

            <div className="compare-card baseline">
              <div className="compare-card-header">
                <span className="compare-card-badge baseline">Static Threshold</span>
                <span className="compare-card-score">
                  {(result.baseline.evaluation.score * 100).toFixed(0)}%
                </span>
              </div>
              <TaskGrid eval={result.baseline.evaluation} />
              <div className="compare-stat">
                <span className="compare-stat-label">Time to Remediate</span>
                <span className="compare-stat-value">
                  {result.baseline.mttr_seconds ? `${result.baseline.mttr_seconds}s` : 'Did not resolve'}
                </span>
              </div>
              <div className="compare-stat">
                <span className="compare-stat-label">Detection Step</span>
                <span className="compare-stat-value">
                  {result.baseline.detection_step > 0 ? `Step ${result.baseline.detection_step}` : 'Not detected'}
                </span>
              </div>
            </div>
          </div>

          <div className="compare-charts">
            <div className="panel">
              <h3>MTTR Comparison</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={mttrData} margin={{ left: 20, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" tick={{ fontSize: 12, fill: '#e2e8f0' }} />
                  <YAxis
                    tick={{ fontSize: 11, fill: '#94a3b8' }}
                    label={{ value: 'seconds', angle: -90, position: 'insideLeft', style: { fill: '#94a3b8', fontSize: 11 } }}
                  />
                  <Tooltip
                    contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6 }}
                    formatter={(v: number | undefined) => [`${v ?? 0}s`, 'MTTR']}
                  />
                  <Bar dataKey="mttr" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                    {mttrData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="panel">
              <h3>Task Accuracy</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={scoreData} margin={{ left: 20, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="task" tick={{ fontSize: 12, fill: '#e2e8f0' }} />
                  <YAxis
                    tick={{ fontSize: 11, fill: '#94a3b8' }}
                    domain={[0, 1]}
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  />
                  <Tooltip
                    contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6 }}
                    formatter={(v: number | undefined) => [`${((v ?? 0) * 100).toFixed(0)}%`, '']}
                  />
                  <Bar dataKey="ml" name="ML Agent" fill="#22c55e" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                  <Bar dataKey="baseline" name="Baseline" fill="#ef4444" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
