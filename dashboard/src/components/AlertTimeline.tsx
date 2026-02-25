interface Alert {
  step: number;
  timestamp: string;
  diagnosis: string;
  service: string;
  action: string;
  confidence: number;
  summary: string;
}

interface AlertTimelineProps {
  alerts: Alert[];
}

export default function AlertTimeline({ alerts }: AlertTimelineProps) {
  if (alerts.length === 0) {
    return (
      <div className="panel">
        <h3>Alert Timeline</h3>
        <p className="text-muted">No alerts yet. Monitoring...</p>
      </div>
    );
  }

  return (
    <div className="panel">
      <h3>Alert Timeline</h3>
      <div className="timeline">
        {alerts.map((alert, idx) => (
          <div key={idx} className={`alert-item alert-${alert.diagnosis.includes('security') || alert.diagnosis === 'brute_force' ? 'security' : 'ops'}`}>
            <div className="alert-header">
              <span className="alert-step">Step {alert.step}</span>
              <span className="alert-type">{alert.diagnosis.replace(/_/g, ' ')}</span>
              <span className="alert-confidence">{(alert.confidence * 100).toFixed(0)}%</span>
            </div>
            <div className="alert-body">
              <strong>{alert.service}</strong> &rarr; {alert.action.replace(/_/g, ' ')}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
