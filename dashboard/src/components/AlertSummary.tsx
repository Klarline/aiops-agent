interface AlertSummaryProps {
  summary: string;
  diagnosis: string;
  severity: string;
}

export default function AlertSummary({ summary, diagnosis, severity }: AlertSummaryProps) {
  const severityClass = severity === 'critical' ? 'severity-critical' :
                        severity === 'high' ? 'severity-high' :
                        severity === 'medium' ? 'severity-medium' : 'severity-low';

  return (
    <div className={`panel summary-card ${severityClass}`}>
      <div className="summary-header">
        <h3>Alert Summary</h3>
        <span className={`severity-badge ${severityClass}`}>{severity}</span>
      </div>
      <div className="summary-type">{diagnosis.replace(/_/g, ' ')}</div>
      <p className="summary-text">{summary}</p>
    </div>
  );
}
