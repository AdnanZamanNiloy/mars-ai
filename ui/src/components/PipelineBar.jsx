const STEPS = ["planner", "search", "summarizer", "critic", "synthesizer"];

export default function PipelineBar({ activeStep }) {
  return (
    <section className="card pipeline-card">
      <h3 className="pipeline-title">Research Pipeline</h3>
      <div className="pipeline-track" role="list" aria-label="Research pipeline steps">
        {STEPS.map((step) => {
          const isActive = activeStep === step;
          return (
            <div key={step} className={`pipeline-step${isActive ? " is-active" : ""}`} role="listitem">
              <span className="pipeline-dot" />
              <span className="pipeline-label">{step}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}
