interface EvalResult {
  detection: boolean;
  localization: boolean;
  diagnosis: boolean;
  mitigation: boolean;
  score: number;
  details?: Record<string, string>;
}

interface EvalBadgeProps {
  result: EvalResult;
  detectionStep: number;
}

export default function EvalBadge({ result, detectionStep }: EvalBadgeProps) {
  const tasks = [
    { key: 'detection', label: 'Detection', pass: result.detection },
    { key: 'localization', label: 'Localization', pass: result.localization },
    { key: 'diagnosis', label: 'Diagnosis', pass: result.diagnosis },
    { key: 'mitigation', label: 'Mitigation', pass: result.mitigation },
  ];

  const scorePercent = (result.score * 100).toFixed(0);

  return (
    <div className="eval-strip">
      <div className="eval-score">
        <span className="eval-score-num">{scorePercent}%</span>
        <span className="eval-score-label">Score</span>
      </div>
      <div className="eval-tasks">
        {tasks.map(t => (
          <div key={t.key} className={`eval-task ${t.pass ? 'pass' : 'fail'}`}>
            <span className="eval-task-icon">{t.pass ? '✓' : '✗'}</span>
            <span className="eval-task-label">{t.label}</span>
          </div>
        ))}
      </div>
      {detectionStep > 0 && (
        <div className="eval-mttr">
          <span className="eval-mttr-value">{detectionStep * 10}s</span>
          <span className="eval-mttr-label">MTTR</span>
        </div>
      )}
      {result.details && (
        <div className="eval-details">
          {result.details.expected_fault && (
            <span className="eval-detail">
              Expected: {result.details.expected_fault.replace(/_/g, ' ')}
            </span>
          )}
          {result.details.agent_diagnosed && (
            <span className="eval-detail">
              Diagnosed: {result.details.agent_diagnosed.replace(/_/g, ' ')}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
