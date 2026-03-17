export default function FinalAnswerCard({ answer, confidence, running }) {
  return (
    <section className="card final-answer-card">
      <header className="final-answer-header">
        <h2>Final Answer</h2>
        {typeof confidence === "number" ? <span className="confidence-pill">{Math.round(confidence * 100)}%</span> : null}
      </header>
      <div className="final-answer-content">
        {answer ? (
          <p>{answer}</p>
        ) : (
          <p className="placeholder-text">
            {running
              ? "Synthesizing evidence..."
              : "Run a query to generate a synthesized final answer."}
          </p>
        )}
      </div>
    </section>
  );
}
