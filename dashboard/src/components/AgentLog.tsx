interface LogEntry {
  step: number;
  timestamp: string;
  diagnosis: string;
  service: string;
  confidence: number;
  action: string;
  summary: string;
  note?: string;
}

interface AgentLogProps {
  log: LogEntry[];
}

export default function AgentLog({ log }: AgentLogProps) {
  return (
    <div className="panel">
      <h3>Agent Reasoning Log</h3>
      {log.length === 0 ? (
        <p className="text-muted">No reasoning steps recorded yet.</p>
      ) : (
        <div className="log-container">
          {log.map((entry, idx) => (
            <div key={idx} className="log-entry">
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
              <div className="log-summary">{entry.summary}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
