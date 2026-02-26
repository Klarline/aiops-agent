import { useRef, useEffect } from 'react';

interface ChainStep {
  step: number;
  thought?: string;
  tool?: string | null;
  args?: Record<string, unknown>;
  observation?: Record<string, unknown>;
}

interface RuleLogEntry {
  mode: 'rule-based';
  step: number;
  timestamp: string;
  diagnosis: string;
  service: string;
  confidence: number;
  action: string;
  summary: string;
  note?: string;
}

interface ReactLogEntry {
  mode: 'react';
  step: number;
  timestamp: string;
  thought: string;
  action: string;
  target: string;
  chain_length: number;
}

type LogEntry = RuleLogEntry | ReactLogEntry;

interface AgentLogProps {
  log: LogEntry[];
  reasoningChain?: ChainStep[];
}

function RuleEntry({ entry }: { entry: RuleLogEntry }) {
  return (
    <div className="log-entry">
      <div className="log-phase">
        <span className="phase observe">OBSERVE</span>
        <span className="log-detail">Step {entry.step} — {entry.service}</span>
      </div>
      <div className="log-phase">
        <span className="phase think">THINK</span>
        <span className="log-detail">
          {entry.diagnosis.replace(/_/g, ' ')} ({(entry.confidence * 100).toFixed(0)}% confidence)
        </span>
      </div>
      <div className="log-phase">
        <span className="phase act">ACT</span>
        <span className="log-detail">{entry.action.replace(/_/g, ' ')}</span>
      </div>
      {entry.summary && (
        <div className="log-summary">{entry.summary}</div>
      )}
    </div>
  );
}

function ReactEntry({ entry, chain }: { entry: ReactLogEntry; chain: ChainStep[] }) {
  return (
    <div className="log-entry log-entry--react">
      <div className="log-header">
        <span className="badge badge--react">LLM ReAct</span>
        <span className="log-detail">
          Step {entry.step} — {entry.chain_length} reasoning steps
        </span>
      </div>

      {chain.length > 0 && (
        <div className="react-chain">
          {chain.map((s, i) => (
            <div key={i} className="react-step">
              {s.thought && (
                <div className="log-phase">
                  <span className="phase think">THOUGHT</span>
                  <span className="log-detail">{s.thought}</span>
                </div>
              )}
              {s.tool && (
                <div className="log-phase">
                  <span className="phase act">ACTION</span>
                  <span className="log-detail log-detail--mono">
                    {s.tool}({s.args ? JSON.stringify(s.args) : ''})
                  </span>
                </div>
              )}
              {s.observation && (
                <div className="log-phase">
                  <span className="phase observe">RESULT</span>
                  <span className="log-detail log-detail--mono">
                    {JSON.stringify(s.observation, null, 0).slice(0, 300)}
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="log-phase">
        <span className="phase act">REMEDIATION</span>
        <span className="log-detail">
          {entry.action.replace(/_/g, ' ')}
          {entry.target ? ` → ${entry.target}` : ''}
        </span>
      </div>
    </div>
  );
}

export default function AgentLog({ log, reasoningChain = [] }: AgentLogProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [log]);

  return (
    <div className="panel agent-log-panel">
      <h3>Agent Reasoning</h3>
      {log.length === 0 ? (
        <p className="text-muted">Monitoring... agent will act when anomalies are confirmed.</p>
      ) : (
        <div className="log-container">
          {log.map((entry, idx) => {
            if (entry.mode === 'react') {
              return (
                <ReactEntry
                  key={idx}
                  entry={entry as ReactLogEntry}
                  chain={idx === log.length - 1 ? reasoningChain : []}
                />
              );
            }
            return <RuleEntry key={idx} entry={entry as RuleLogEntry} />;
          })}
          <div ref={endRef} />
        </div>
      )}
    </div>
  );
}
