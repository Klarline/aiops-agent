import { useState, useEffect, useCallback, useRef } from 'react';
import MetricsPanel from './components/MetricsPanel';
import ShapExplainer from './components/ShapExplainer';
import AgentLog from './components/AgentLog';
import IncidentComparison from './components/IncidentComparison';
import EvalBadge from './components/EvalBadge';

const API = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/agent/ws';

interface StepData {
  step: number;
  metrics: Record<string, Record<string, number>>;
  anomaly_scores: Record<string, number>;
  action: string;
  target?: string;
  diagnosis?: string;
  explanation?: string;
  confidence?: number;
}

interface ShapData {
  service: string;
  features: { name: string; value: number; human_label?: string }[];
  all_values?: Record<string, number>;
}

interface EvalResult {
  detection: boolean;
  localization: boolean;
  diagnosis: boolean;
  mitigation: boolean;
  score: number;
  details?: Record<string, string>;
}

type TabId = 'live' | 'compare';

export default function App() {
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [selectedScenario, setSelectedScenario] = useState('');
  const [selectedService, setSelectedService] = useState('api-gateway');
  const [services, setServices] = useState<string[]>([]);
  const [tab, setTab] = useState<TabId>('live');

  const [metricsTimeline, setMetricsTimeline] = useState<StepData[]>([]);
  const [agentLog, setAgentLog] = useState<any[]>([]);
  const [reasoningChain, setReasoningChain] = useState<any[]>([]);
  const [shapData, setShapData] = useState<ShapData | null>(null);
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null);
  const [detectionStep, setDetectionStep] = useState(-1);

  const [streaming, setStreaming] = useState(false);
  const [status, setStatus] = useState('Select a scenario to begin');
  const [currentStep, setCurrentStep] = useState(0);
  const [speed, setSpeed] = useState(60);

  const wsRef = useRef<WebSocket | null>(null);
  const abortRef = useRef(false);

  useEffect(() => {
    fetch(`${API}/agent/scenarios`)
      .then(r => r.json())
      .then(d => setScenarios(d.scenarios || []))
      .catch(() => {});
  }, []);

  const resetState = useCallback(() => {
    setMetricsTimeline([]);
    setAgentLog([]);
    setReasoningChain([]);
    setShapData(null);
    setEvalResult(null);
    setDetectionStep(-1);
    setCurrentStep(0);
    abortRef.current = false;
  }, []);

  const runScenarioWs = useCallback(() => {
    if (!selectedScenario) return;
    resetState();
    setStreaming(true);
    setStatus('Connecting...');

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({
        scenario_id: selectedScenario,
        speed_ms: speed,
      }));
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);

      if (msg.type === 'init') {
        setServices(msg.services);
        setStatus(`Running: ${msg.description}`);
      } else if (msg.type === 'step') {
        setCurrentStep(msg.step);
        setMetricsTimeline(prev => [...prev, msg as StepData]);
      } else if (msg.type === 'detection') {
        setCurrentStep(msg.step);
        setDetectionStep(msg.step);
        setMetricsTimeline(prev => [...prev, msg as StepData]);
        if (msg.shap) setShapData(msg.shap);
        if (msg.log) setAgentLog(msg.log);
        if (msg.reasoning_chain) setReasoningChain(msg.reasoning_chain);
        setStatus(`Detected: ${msg.diagnosis?.replace(/_/g, ' ')} on ${msg.target}`);
      } else if (msg.type === 'complete') {
        setStreaming(false);
        if (msg.evaluation) setEvalResult(msg.evaluation);
        if (msg.detection_step > 0) setDetectionStep(msg.detection_step);
        setStatus('Scenario complete');
      } else if (msg.type === 'error') {
        setStreaming(false);
        setStatus(`Error: ${msg.message}`);
      }
    };

    ws.onerror = () => {
      setStreaming(false);
      setStatus('WebSocket error — falling back to HTTP');
      runScenarioHttp();
    };

    ws.onclose = () => {
      setStreaming(false);
    };
  }, [selectedScenario, speed, resetState]);

  const runScenarioHttp = useCallback(async () => {
    if (!selectedScenario) return;
    resetState();
    setStreaming(true);
    setStatus('Starting scenario...');

    try {
      const startRes = await fetch(`${API}/agent/scenarios/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario_id: selectedScenario }),
      });
      const startData = await startRes.json();
      setServices(startData.services || []);
      setStatus(`Running: ${startData.description}`);

      let running = true;
      while (running && !abortRef.current) {
        const res = await fetch(`${API}/agent/step?n_steps=5`, { method: 'POST' });
        const data = await res.json();
        running = data.running;

        if (data.steps) {
          setMetricsTimeline(prev => [...prev, ...data.steps]);
          const lastStep = data.steps[data.steps.length - 1];
          setCurrentStep(lastStep.step);
        }
        if (data.detection_step > 0) setDetectionStep(data.detection_step);

        const logRes = await fetch(`${API}/agent/log`);
        const logData = await logRes.json();
        if (logData.log?.length > 0) setAgentLog(logData.log);
        if (logData.reasoning_chain?.length > 0) setReasoningChain(logData.reasoning_chain);

        const shapRes = await fetch(`${API}/agent/shap`);
        const shapJson = await shapRes.json();
        if (shapJson.features?.length > 0) setShapData(shapJson);

        await new Promise(r => setTimeout(r, speed));
      }

      const evalRes = await fetch(`${API}/agent/evaluate`);
      const evalData = await evalRes.json();
      setEvalResult(evalData);
      setStatus('Scenario complete');
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
    } finally {
      setStreaming(false);
    }
  }, [selectedScenario, speed, resetState]);

  const stopScenario = useCallback(() => {
    abortRef.current = true;
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStreaming(false);
    setStatus('Stopped');
  }, []);

  const latestMetrics: Record<string, Record<string, number>> =
    metricsTimeline.length > 0
      ? metricsTimeline[metricsTimeline.length - 1].metrics
      : {};

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <h1>AIOps Agent</h1>
          <span className="subtitle">Autonomous Cloud Operations</span>
        </div>
        <div className="header-tabs">
          <button className={`tab ${tab === 'live' ? 'active' : ''}`} onClick={() => setTab('live')}>
            Live Demo
          </button>
          <button className={`tab ${tab === 'compare' ? 'active' : ''}`} onClick={() => setTab('compare')}>
            Agent vs Baseline
          </button>
        </div>
      </header>

      {tab === 'live' ? (
        <>
          <div className="controls-bar">
            <select
              value={selectedScenario}
              onChange={e => setSelectedScenario(e.target.value)}
              className="scenario-select"
            >
              <option value="">Select scenario...</option>
              {scenarios.map(s => (
                <option key={s} value={s}>{s.replace(/_/g, ' ')}</option>
              ))}
            </select>

            <div className="speed-control">
              <label>Speed</label>
              <input
                type="range" min={10} max={200} value={speed}
                onChange={e => setSpeed(Number(e.target.value))}
              />
              <span>{speed}ms</span>
            </div>

            {!streaming ? (
              <button className="btn btn-primary" onClick={runScenarioWs} disabled={!selectedScenario}>
                Run Full Scenario
              </button>
            ) : (
              <button className="btn btn-danger" onClick={stopScenario}>
                Stop
              </button>
            )}

            <select
              value={selectedService}
              onChange={e => setSelectedService(e.target.value)}
              className="service-select"
            >
              {(services.length > 0 ? services : ['api-gateway', 'auth-service', 'order-service', 'user-db', 'order-db']).map(s =>
                <option key={s} value={s}>{s}</option>
              )}
            </select>

            <div className="step-indicator">
              Step {currentStep}
              {detectionStep > 0 && (
                <span className="detection-badge">Detected @ {detectionStep}</span>
              )}
            </div>
          </div>

          <div className="status-bar">
            <span className={`status-dot ${streaming ? 'live' : detectionStep > 0 ? 'detected' : 'idle'}`} />
            {status}
          </div>

          {evalResult && <EvalBadge result={evalResult} detectionStep={detectionStep} />}

          <div className="dashboard-grid">
            <div className="grid-metrics">
              <MetricsPanel
                timeline={metricsTimeline}
                selectedService={selectedService}
                allServices={services.length > 0 ? services : ['api-gateway', 'auth-service', 'order-service', 'user-db', 'order-db']}
                detectionStep={detectionStep}
                latestMetrics={latestMetrics}
              />
            </div>

            <div className="grid-sidebar">
              <ShapExplainer data={shapData} />
              <AgentLog log={agentLog} reasoningChain={reasoningChain} />
            </div>
          </div>
        </>
      ) : (
        <IncidentComparison
          scenarios={scenarios}
          apiBase={API}
        />
      )}
    </div>
  );
}
