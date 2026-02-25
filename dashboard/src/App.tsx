import { useState, useEffect, useCallback } from 'react';
import MetricsPanel from './components/MetricsPanel';
import AlertTimeline from './components/AlertTimeline';
import ShapExplainer from './components/ShapExplainer';
import AlertSummary from './components/AlertSummary';
import AgentLog from './components/AgentLog';

const API = 'http://localhost:8000';

function App() {
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [selectedScenario, setSelectedScenario] = useState('');
  const [selectedService, setSelectedService] = useState('api-gateway');
  const [services, setServices] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<Record<string, Record<string, number>>>({});
  const [agentLog, setAgentLog] = useState<any[]>([]);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState('idle');
  const [evalResult, setEvalResult] = useState<any>(null);

  useEffect(() => {
    fetch(`${API}/agent/scenarios`)
      .then(r => r.json())
      .then(d => setScenarios(d.scenarios || []))
      .catch(() => {});
  }, []);

  const startScenario = useCallback(async () => {
    if (!selectedScenario) return;
    setStatus('starting...');
    setAgentLog([]);
    setEvalResult(null);
    const res = await fetch(`${API}/agent/scenarios/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario_id: selectedScenario }),
    });
    const data = await res.json();
    setServices(data.services || []);
    setRunning(true);
    setStatus(`Running: ${data.description}`);
  }, [selectedScenario]);

  const runSteps = useCallback(async (n: number) => {
    const res = await fetch(`${API}/agent/step?n_steps=${n}`, { method: 'POST' });
    const data = await res.json();
    setRunning(data.running);

    const metricsRes = await fetch(`${API}/metrics/current`);
    const metricsData = await metricsRes.json();
    setMetrics(metricsData);

    const logRes = await fetch(`${API}/agent/log`);
    const logData = await logRes.json();
    setAgentLog(logData.log || []);

    if (!data.running) setStatus('Scenario complete');
  }, []);

  const evaluate = useCallback(async () => {
    const res = await fetch(`${API}/agent/evaluate`);
    const data = await res.json();
    setEvalResult(data);
  }, []);

  const latestAlert = agentLog.length > 0 ? agentLog[agentLog.length - 1] : null;

  return (
    <div className="app">
      <header className="app-header">
        <h1>AIOps Agent Dashboard</h1>
        <span className="subtitle">AIOpsLab-aligned Autonomous Agent</span>
      </header>

      <div className="controls">
        <select value={selectedScenario} onChange={e => setSelectedScenario(e.target.value)}>
          <option value="">Select scenario...</option>
          {scenarios.map(s => <option key={s} value={s}>{s.replace(/_/g, ' ')}</option>)}
        </select>
        <button onClick={startScenario} disabled={!selectedScenario}>Start</button>
        <button onClick={() => runSteps(10)} disabled={!running}>Run 10 Steps</button>
        <button onClick={() => runSteps(50)} disabled={!running}>Run 50 Steps</button>
        <button onClick={evaluate}>Evaluate</button>
        <select value={selectedService} onChange={e => setSelectedService(e.target.value)}>
          {(services.length > 0 ? services : ['api-gateway', 'auth-service', 'order-service', 'user-db', 'order-db']).map(s =>
            <option key={s} value={s}>{s}</option>
          )}
        </select>
      </div>

      <div className="status-bar">{status}</div>

      {evalResult && (
        <div className="panel eval-panel">
          <h3>Evaluation (AIOpsLab Tasks)</h3>
          <div className="eval-grid">
            <div className={`eval-item ${evalResult.detection ? 'pass' : 'fail'}`}>
              Detection: {evalResult.detection ? 'PASS' : 'FAIL'}
            </div>
            <div className={`eval-item ${evalResult.localization ? 'pass' : 'fail'}`}>
              Localization: {evalResult.localization ? 'PASS' : 'FAIL'}
            </div>
            <div className={`eval-item ${evalResult.diagnosis ? 'pass' : 'fail'}`}>
              Diagnosis: {evalResult.diagnosis ? 'PASS' : 'FAIL'}
            </div>
            <div className={`eval-item ${evalResult.mitigation ? 'pass' : 'fail'}`}>
              Mitigation: {evalResult.mitigation ? 'PASS' : 'FAIL'}
            </div>
          </div>
        </div>
      )}

      <div className="main-grid">
        <MetricsPanel data={metrics} selectedService={selectedService} />
        {latestAlert && (
          <AlertSummary
            summary={latestAlert.summary}
            diagnosis={latestAlert.diagnosis}
            severity={latestAlert.confidence > 0.8 ? 'critical' : latestAlert.confidence > 0.5 ? 'high' : 'medium'}
          />
        )}
        <ShapExplainer features={latestAlert ? [
          { name: 'feature_1', value: latestAlert.confidence },
          { name: 'feature_2', value: latestAlert.confidence * 0.7 },
          { name: 'feature_3', value: latestAlert.confidence * 0.4 },
        ] : []} />
        <AlertTimeline alerts={agentLog} />
        <AgentLog log={agentLog} />
      </div>
    </div>
  );
}

export default App;
