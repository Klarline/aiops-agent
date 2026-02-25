import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface ShapFeature {
  name: string;
  value: number;
}

interface ShapExplainerProps {
  features: ShapFeature[];
}

export default function ShapExplainer({ features }: ShapExplainerProps) {
  if (features.length === 0) {
    return (
      <div className="panel">
        <h3>SHAP Explanation</h3>
        <p className="text-muted">Select an alert to see feature attributions.</p>
      </div>
    );
  }

  const data = features.map((f) => ({
    name: f.name.replace(/_/g, ' ').slice(0, 20),
    value: f.value,
    fill: f.value > 0 ? '#ef4444' : '#3b82f6',
  }));

  return (
    <div className="panel">
      <h3>SHAP Feature Attribution</h3>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} layout="vertical" margin={{ left: 100 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis type="category" dataKey="name" width={100} />
          <Tooltip />
          <Bar dataKey="value">
            {data.map((entry, index) => (
              <Cell key={index} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
